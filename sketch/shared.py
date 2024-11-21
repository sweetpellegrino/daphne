import os
import subprocess
import json
import pandas as pd
from tabulate import tabulate
import psutil
import time
from multiprocessing.pool import ThreadPool
import io

#------------------------------------------------------------------------------
# RUN COMMAND
#------------------------------------------------------------------------------

DAPHNE_ENV = {
    "LD_LIBRARY_PATH": "$PWD/lib:$PWD/thirdparty/installed/lib:$LD_LIBRARY_PATH"
}

def poll_memory_info(process, process_memory, poll_interval=0.001):
    peak_rss = 0
    peak_vms = 0
    while process.poll() is None: 
        rss = process_memory.memory_info().rss // 1024  # Memory usage in KB
        vms = process_memory.memory_info().vms // 1024  # Memory usage in KB
        if rss > peak_rss:
            peak_rss = rss

        if vms > peak_vms:
            peak_vms = vms
        time.sleep(poll_interval)
    return peak_rss, peak_vms

    
def run_command(command, cwd, env): 

    pool = ThreadPool(processes=1)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env={**env, **os.environ, **DAPHNE_ENV})

    process_memory = psutil.Process(process.pid)
    async_memory = pool.apply_async(poll_memory_info, (process, process_memory))

    buffer = io.BytesIO();
    with process.stderr:
        for line in iter(process.stderr.readline, ''):
            if line:
                buffer.write(line)
            else:
                break

    process.stderr.close()

    process.wait()

    peak_rss, peak_vms = async_memory.get()
    stdout, _ = process.communicate()

    buffer.seek(0)
    return peak_rss, peak_vms, stdout.decode(), buffer.read().decode()

def runner(args, cmd, cwd):

    tool_env = TOOLS[args.tool]["ENV"]
    env_str = " ".join(f"{k}=\"{v}\"" for k, v in tool_env.items())
    cmd_str = " ".join(cmd)
    print(f"Run: {env_str} {cmd_str} {cwd}")

    timings = []
    for i in range(0, args.samples):

        peak_rss, peak_vms, stdout, stderr = run_command(cmd, cwd, tool_env)

        if args.verbose_output:
            print(stdout)
            print(stderr)
            print(f"Peak RSS usage: {peak_rss} KB")
            print(f"Peak VMS usage: {peak_vms} KB")

        if args.tool == "MALLOC":
            timing = json.loads(stderr.split("\n")[-4])
            timing["tool"] = TOOLS[args.tool]["GET_INFO"](stdout,stderr)
        else:
            timing = json.loads(stderr.split("\n")[-2])
            timing["tool"] = TOOLS[args.tool]["GET_INFO"](stdout)

        timing["peak_rss_kilobytes"] = peak_rss
        timing["peak_vms_kilobytes"] = peak_vms

        df = pd.json_normalize(timing, sep=".")
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
        timings.append(timing)

    return timings



#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------

def extract_f1xm3(stdout):
    lines = stdout.split('\n')

    for line in reversed(lines):
        if "F1XM3" in line:
            number = line.split("F1XM3:")[1]
            return int(number)
    return None

def extract_malloc_f1xm3(stdout,stderr):
    lines = stdout.split('\n')

    f1xm3 = -1
    for line in reversed(lines):
        print(line)
        if "F1XM3" in line:
            number = line.split("F1XM3:")[1]
            f1xm3 = int(number)
            break

    lines = stderr.split('\n')
    total_alloc = -1
    for line in lines:
        if "ALLOC" in line:
            alloc = line.split("ALLOC:")[1] 
            total_alloc += int(alloc)
            
    return {
            "time": f1xm3, 
            "tmalloc": total_alloc
        }

def extract_papi(stdout):
    lines = stdout.split('\n')

    offset = 0
    for i, line in enumerate(lines):
        if line.startswith("PAPI-HL Output:"):
           offset = i
           break
    t = "".join(lines[offset+1:])
    j = json.loads(t)
    out = j["threads"]["0"]["regions"]["0"]
    del out["name"]
    del out["parent_region_id"]
    return out

TOOLS = {
    "PAPI_STD": {
        "ENV": {
            "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES",
            "PAPI_REPORT": "1"
        },
        "START_OP": "startProfiling();",
        "STOP_OP": "stopProfiling();",
        "END_OP": "",
        "GET_INFO": extract_papi
    },
    "PAPI_L1": {
        "ENV": {
            "PAPI_EVENTS": "perf::L1-dcache-load-misses,perf::L1-dcache-loads,perf::L1-dcache-prefetches,perf::L1-icache-load-misses,perf::L1-icache-loads",
            "PAPI_REPORT": "1",
        },
        "START_OP": "startProfiling();",
        "STOP_OP": "stopProfiling();",
        "END_OP": "",
        "GET_INFO": extract_papi
    },
    "PAPI_MPLX": {
        "ENV": {
            "PAPI_EVENTS": "perf::CYCLES,perf::INSTRUCTIONS,perf::CACHE-REFERENCES,perf::CACHE-MISSES,perf::BRANCHES,perf::BRANCH-MISSES,perf::L1-dcache-load-misses,perf::L1-dcache-loads,perf::L1-dcache-prefetches,perf::L1-icache-load-misses,perf::L1-icache-loads",
            "PAPI_REPORT": "1",
            "PAPI_MULTIPLEX": "1",
        },
        "START_OP": "startProfiling();",
        "STOP_OP": "stopProfiling();",
        "END_OP": "",
        "GET_INFO": extract_papi
    },
    "NOW": {
        "ENV": {},
        "START_OP": "start = now();",
        "STOP_OP": "end = now();",
        "END_OP": "print(\"F1XM3:\"+ (end - start));",
        "GET_INFO": extract_f1xm3
    },
    "MALLOC": {
        "ENV": {
            "LD_PRELOAD": "../sketch/prof_malloc/lib_prof_malloc.so"
        },
        "START_OP": "start = now();",
        "STOP_OP": "end = now();",
        "END_OP": "print(\"F1XM3:\"+ (end - start));",
        "GET_INFO": extract_malloc_f1xm3
    }
}
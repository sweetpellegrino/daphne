# gray 
# (245,245,245) : #f5f5f5
# (102,102,102) : #666666 (0.40, 0.40, 0.40)
# blue
# (219,232,251) : #dbe8fb
# (110,143,189) : #6e8fbd (0.43, 0.56, 0.74)
# green
# (214,232,213) : #d6e8d5
# (131,178,106) : #83b26a (0.51, 0.70, 0.42)
# orange
# (255,230,206) : #ffe6ce
# (214,154,35) : #d69a23 (0.84, 0.60, 0.14)
# red
# (247,206,206) : #f7cece
# (182,85,82) : #b65552 (0.71, 0.33, 0.32)
# purple
# (225,214,231) :#e1d6e7
# (150,116,165) :#9674a5 (0.59, 0.45, 0.65)

figsize_width = 5
figsize_height = 3.5
bar_width = 0.45
font_size = 11
offset_max = 0.2

colors = ["#f5f5f5", "#dbe8fb", "#d6e8d5", "#e1d6e7", "#ffe6ce", "#f7cece", "#cef7f7"]
edgecolors = ["#666666", "#6e8fbd", "#83b26a", "#9674a5", "#d69a23", "#b65552", "#52b3b6"]

xticks_name = ["Base", "BVec", "CTB #1", "CTB #2", "CTB #3"]
xticks_shortname = ["B", "BV", "C1", "C2", "C3"]

units = {
    "exec_time": {
        "label": "(Mean) Execution Time [s]",
        "conversion": 1e9
    },
     "comp_time": {
        "label": "(Mean) Compilation Time [s]",
        "conversion": 1
    }
}
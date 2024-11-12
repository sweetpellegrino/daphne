#bin/bash

./pack.sh --version vec
scp  -i ~/.ssh/dev-gp-hetzner ./build/daphne-X86-64-vec-bin.tgz root@142.132.190.42:~/

ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'rm -fr daphne-X86-64-vec-bin'
ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'rm -fr sketch/'
ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'rm -fr run_*.py'
ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'rm -fr shared.py'

ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'tar -xvf daphne-X86-64-vec-bin.tgz'

ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'mv daphne-X86-64-vec-bin/sketch .'
ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'mv sketch/run_*.py .'
ssh -i ~/.ssh/dev-gp-hetzner root@142.132.190.42 'mv sketch/shared.py .'

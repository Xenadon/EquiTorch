import subprocess
from time import sleep

def run_forwards(i, N, channel, L):
    L2 = (L[0], L[1]*2)
    print(i+1,flush=True)
    subprocess.run(
        ['python', '-c', f'from test_so3 import *; check_forward_single({i}, {N}, {L}, {L}, {channel},{channel})'])
    # sleep(1)

def run_backwards(i, N, channel, L):
    L2 = (L[0], L[1]*2)
    print(i+1,flush=True)
    subprocess.run(
        ['python', '-c', f'from test_so3 import *; check_backward_single({i}, {N}, {L}, {L}, {channel},{channel})'])
    # sleep(1)

convs = 7

rep = 1

# for N in [200]:
#     for channel in [64]:
#         for L in [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8)]:
#             print('- Check forward performance', flush=True)
#             print(f'- setting: N={N}, L={L}, L1={L}, L2={L}, in={channel}, out={channel}', flush=True)
#             print('-'*100,flush=True)
#             for i in range(convs):
#                 if i == 2:
#                     continue
#                 for j in range(rep):
#                     run_forwards(i, N,channel,L)
#             print('-'*100,flush=True)

for N in [200]:
    for channel in [64]:
        for L in [(0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8)]:
            print('- Check backward performance', flush=True)
            print(f'- setting: N={N}, L={L}, L1={L}, in={channel}, out={channel}', flush=True)
            print('-'*100,flush=True)
            for i in range(convs):
                if i == 2:
                    continue
                for j in range(rep):
                    run_backwards(i, N,channel,L)
            print('-'*100,flush=True)


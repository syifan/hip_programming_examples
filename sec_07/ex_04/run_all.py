import os
import re
import subprocess

def generate_code(num_reg):
    with open('main.cpp', 'r') as f:
        code = f.read()

    code = code.replace('/*LDS_SIZE*/', str(num_reg))

    with open('main_{reg}.cpp'.format(reg=num_reg), 'w') as f:
        f.write(code)

def compile(reg):
    os.system('hipcc main_{reg}.cpp -o main_{reg}'.format(reg=reg))

def run(reg):
    out = subprocess.check_output([
        './main_{reg}'.format(reg=reg), '1',
    ])
    out = out.decode('utf-8')

    time = re.search(r'Time: (\d+\.\d+) ms', out).group(1)
    return float(time)
    

def profile(reg):
    os.system('rocprof -i profiler_input.txt -o profiler_output_{reg}.csv ./main_{reg}'.format(reg=reg))

    ## Get the second row of profiler_output_{reg}.csv
    with open('profiler_output_{reg}.csv'.format(reg=reg), 'r') as f:
        lines = f.readlines()
        row = lines[1]
        row = row.split(',')
        row = row[1:]

    row = ','.join(row)
    return row

def main():
    with open('times.csv', 'w') as f:
        f.write("num_reg,time,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,arch_vgpr,accum_vgpr,sgpr,wave_size,sig,obj,Wavefronts,VALUInsts,SALUInsts,SFetchInsts,GDSInsts,VALUBusy,MemUnitStalled,FetchSize,WriteSize\n")

        independent = [] \
            + list(range(32, 64, 2)) \
            + list(range(64, 128, 4)) \
            + list(range(128, 256, 8)) \
            + list(range(256, 512, 16)) \
            + list(range(512, 1024, 32)) \
            + list(range(1024, 2048, 64)) \
            + list(range(2048, 4096, 128)) \
            + list(range(4096, 8192, 256)) \
            + list(range(8192, 16384, 512)) \
            + list(range(16384, 32768, 1024)) 
            
        

        for i in independent:
            print(f'Running num register {i}')

            generate_code(i)
            compile(i)
            time = run(i)
            row = profile(i)
            print(time, row)

            f.write(f'{i},{time},')
            f.write(row)

            print(time, row)

    
            



if __name__ == '__main__':
    main()
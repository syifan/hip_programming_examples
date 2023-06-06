import os
import re
import subprocess

def generate_code(unroll):
    part1 = "float value{unroll} = d_image[idx + {unroll}];"
    part2 = "value{unroll} = (uint8_t)(pow(value{unroll} / 255.0, gamma) * 255.0);"
    part3 = "d_image[idx + {unroll}] = value{unroll};"
    
    ## Load main.cpp as string
    with open('main.cpp', 'r') as f:
        code = f.read()

    ## Generate the unrolled code
    code = code.replace('/*[unroll_factor]*/', str(unroll))

    part1_code = '\n'.join([part1.format(unroll=i) for i in range(unroll)])
    part2_code = '\n'.join([part2.format(unroll=i) for i in range(unroll)])
    part3_code = '\n'.join([part3.format(unroll=i) for i in range(unroll)])

    code = code.replace('/*region1*/', part1_code)
    code = code.replace('/*region2*/', part2_code)
    code = code.replace('/*region3*/', part3_code)

    ## Save the code to main_{unroll}.cpp
    with open('main_{unroll}.cpp'.format(unroll=unroll), 'w') as f:
        f.write(code)

def compile(unroll):
    os.system('hipcc main_{unroll}.cpp -o main_{unroll}'.format(unroll=unroll))

def run(unroll):
    out = subprocess.check_output([
        './main_{unroll}'.format(unroll=unroll), '1',
    ])
    out = out.decode('utf-8')

    time = re.search(r'Time: (\d+\.\d+) ms', out).group(1)
    return float(time)
    

def profile(unroll):
    ## Run rocprof -i profiler_input.txt -o profiler_output_{unroll}.csv main_{unroll}

    os.system('rocprof -i profiler_input.txt -o profiler_output_{unroll}.csv ./main_{unroll}'.format(unroll=unroll))

    ## Get the second row of profiler_output_{unroll}.csv
    with open('profiler_output_{unroll}.csv'.format(unroll=unroll), 'r') as f:
        lines = f.readlines()
        row = lines[1]
        row = row.split(',')
        row = row[1:]

    row = ','.join(row)
    return row

def main():
    with open('times.csv', 'w') as f:
        for i in range(1, 128):
            print(f'Running unrolling {i}')

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
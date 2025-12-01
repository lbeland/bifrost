#!/usr/bin/env python3
import numpy as np

def accumulate(data):
    nframe = 2
    odatas = []
    frame_count = 0
    odata = np.zeros_like(data[0])

    for idx, a in enumerate(data):
        beta = 0.0 if frame_count == 0 else 1.0
        odata = beta * odata + a
        frame_count += 1
        if frame_count == nframe:
            odatas.append(odata)
            frame_count = 0
            # next accumulation starts; first incoming frame will use beta=0 so odata will be overwritten
    return odatas.tolist()

def write_file():
    filename = "test_f32.bin"
    gulp_size = 10  # number of float32 elements per gulp
    n_gulps = 30      # total gulps to write (change as desired)

    data = np.array(list(range(gulp_size * n_gulps)), dtype=np.float32)
    data.tofile(filename)
    print(f"Wrote {data.size} float32 elements to {filename}")

def read_file_input():
    # Edit these constants as needed
    FILENAME = "test_f32.bin"   # file to read
    DTYPE = np.float32       # numpy dtype (e.g. np.float32, np.complex64, np.int16, np.uint8)
    GULP_SIZE = 1024         # number of elements per gulp
    try:
        data = np.fromfile(FILENAME, dtype=DTYPE, count=GULP_SIZE)
        return data
    except Exception as e:
        print("Error reading file:", e)
        return
    
def read_file_output():
    # Edit these constants as needed
    FILENAME = "test_f32.bin.out"   # file to read
    DTYPE = np.float32       # numpy dtype (e.g. np.float32, np.complex64, np.int16, np.uint8)
    GULP_SIZE = 1024         # number of elements per gulp
    try:
        data = np.fromfile(FILENAME, dtype=DTYPE, count=GULP_SIZE)
        return data
    except Exception as e:
        print("Error reading file:", e)
        return

def main():
    # Read one gulp from file

    write_file()

    # data = read_file_input()
    # print("Input Data:", data)

    # # output = accumulate(data)
    # # print("Final Output Data:", output)

    # file_output = read_file_output()
    # print("File Output Data:", file_output[0:20])

if __name__ == "__main__":
    main()
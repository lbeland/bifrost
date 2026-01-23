import os
import numpy as np
import argparse

from copy import deepcopy
import time
import bifrost as bf

from bifrost.pipeline import TransformBlock


class Producer(bf.pipeline.SourceBlock):
    """Generate a sine wave as a SourceBlock.

    Args:
        sourcenames: list of source names (one entry per sequence)
        sampling_rate: samples per second (Hz)
        frequency: sine frequency (Hz)
        n_samples: optional maximum number of samples to produce (None for infinite)
        gulp_nframe: number of frames per gulp
    """
    def __init__(self, sourcenames, sampling_rate=10000, n_messages=1000, n_channels=1, msg_size=1, output_file=None, *args, **kwargs):
        super(Producer, self).__init__(sourcenames, gulp_nframe=1, *args, **kwargs)
        self.sampling_rate = float(sampling_rate)
        self.PERIOD_S = 1.0 / self.sampling_rate
        self.n_samples = int(n_messages)
        self.n_channels = n_channels
        self.msg_size = msg_size
        self.n_emitted = 0
        self.send_times = []
        self.default_data = np.ones((n_channels, msg_size), dtype=np.float64)
        self.next_deadline = None
        self.output_file = output_file

    def create_reader(self, sourcename):
        # Simple context manager that yields self as the reader object
        from contextlib import contextmanager
        @contextmanager
        def _reader():
            yield self
        return _reader()
    
    def on_sequence(self, ireader, filename):
        ohdr = {
            'name': filename,
            '_tensor': {
                'dtype':  'f64',
                'shape':  [-1, self.n_channels, self.msg_size],
                'labels':  ['streamed', 'gulped'],
                'units': [None, None],
                'scales': [[0, 1], [0, 1]]
            },
        }
        return [ohdr]

    def on_data(self, reader, ospans):
        # Fill the output span with sine samples, sleeping between samples
        now = time.monotonic()
        if not self.next_deadline:
            self.next_deadline = now
        if now >= self.next_deadline:
            ospan = ospans[0]
            
            data = self.default_data * now
            # print(self.n_emitted, data.shape, data)
            ospan.data[:] = data
            self.n_emitted += 1
            self.send_times.append(now)

            self.next_deadline += self.PERIOD_S
            # Stop if we've emitted enough samples (first is omitted from timing)
            if (self.n_emitted >= self.n_samples + 1):
                self.shutdown_event.set()
            return [1]
        else:
            return [0]
    def shutdown(self) -> None:
        print(f"Total messages sent: {len(self.send_times)}")

        self.send_times = self.send_times[1:]
        send_times_diff = np.diff(self.send_times)
        print(f"Average send period (s): {np.mean(send_times_diff):.6f}")
        print(f"Max send period (s): {np.max(send_times_diff):.6f}, idx: {np.argmax(send_times_diff)}")
        print(f"Std send period (s): {np.std(send_times_diff):.6f}")
        save = np.vstack((self.send_times, np.append(send_times_diff, 0)))
        np.savetxt(self.output_file, save.T, delimiter=',', header='send_time,send_period')
        

class Consumer(TransformBlock):
    def __init__(self, iring, gulp_nframe=1,output_file=None, 
             *args, **kwargs):
        super(Consumer, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.recv_times = []
        self.process_times = []
        self.n_recv = 0
        self.output_file = output_file
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        # print(f"Consumer got data gulp: {idata.shape}, {idata}")
        now = time.monotonic()
        self.recv_times.append(now)
        self.process_times.append(now - idata[0,0,0])

        self.n_recv += 1
        return None
    def shutdown(self) -> None:
        print(f"Total messages processed: {len(self.process_times)}")
        process_times = self.process_times[1:]     # exclude first measuremnt because of startup delay
        # print(f"Average processing time (s): {np.mean(process_times)/1e9:.6f}")
        # print(f"Max processing time (s): {np.max(process_times)/1e9:.6f}, idx: {np.argmax(process_times)}")
        # print(f"Std processing time (s): {np.std(process_times)/1e9:.6f}")
        # print('\n')
        recv_times = self.recv_times[1:]
        recv_times_diff = np.diff(recv_times)
        print(f"Average receive period (s): {np.mean(recv_times_diff):.6f}")
        print(f"Max receive period (s): {np.max(recv_times_diff):.6f}, idx: {np.argmax(recv_times_diff)}")
        print(f"Std receive period (s): {np.std(recv_times_diff):.6f}")

        save = np.vstack((recv_times, process_times, np.append(recv_times_diff, 0)))
        np.savetxt(self.output_file, save.T, delimiter=',', header='recv_time,latency,recv_period')
        self.shutdown_event.set()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_messages", type=int, default=1000)
    parser.add_argument("--freq", type=int, default=2500)
    parser.add_argument("--msg_size", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--output_file", type=str, default="send_times.csv")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    if os.path.exists(f"results/{args.freq}_{args.num_channels}_{args.msg_size}_recv_times.csv"):
        return  # Skip if already exists

    data1 = Producer(['random'], sampling_rate=args.freq, n_messages=args.n_messages, n_channels=args.num_channels, msg_size=args.msg_size,output_file=f"results/{args.freq}_{args.num_channels}_{args.msg_size}_send_times.csv")

    data2 = Consumer(data1, output_file=f"results/{args.freq}_{args.num_channels}_{args.msg_size}_recv_times.csv", gulp_nframe=1)

    pipeline = bf.get_default_pipeline()
    # pipeline.dot_graph().render(filename='benchmark', format='png', cleanup=True)

    pipeline.run()

    pipeline.shutdown()
    print(f"Benchmark complete")

if __name__ == "__main__":
    main()

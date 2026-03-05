import os
import numpy as np
import pandas as pd
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
    def __init__(self, sourcenames, n_messages=1000, n_channels=1, msg_size=1, output_file=None, *args, **kwargs):
        super(Producer, self).__init__(sourcenames, gulp_nframe=1, *args, **kwargs)
        self.n_samples = int(n_messages)
        self.n_channels = n_channels
        self.msg_size = msg_size
        self.n_emitted = 0
        self.send_times = []
        self.default_data = np.ones((n_channels, msg_size), dtype=np.float64)
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
        now = time.monotonic()
        ospan = ospans[0]
        data = self.default_data * now
        ospan.data[:] = data
        # print(self.n_emitted, data.shape, data)
        if self.n_emitted < 500:
            self.n_emitted += 1
            return [1]  # warmup phase, don't record times
        else:
            self.send_times.append(now)
            self.n_emitted += 1
            # Stop if we've emitted enough samples
            if (self.n_emitted >= self.n_samples + 500):
                self.shutdown_event.set()
            return [1]
        
    def shutdown(self) -> None:
        print(f"Total messages sent: {len(self.send_times)}")

        send_times_diff = np.diff(self.send_times)
        mean_send_period = np.mean(send_times_diff)*1e6
        std_send_period = np.std(send_times_diff)*1e6
        max_send_period = np.max(send_times_diff)*1e6
        print(f"Average send period (us): {mean_send_period:.4f}")
        print(f"Max send period (us): {max_send_period:.4f}, idx: {np.argmax(send_times_diff)}")
        print(f"Std send period (us): {std_send_period:.4f}")

        results = pd.DataFrame({"Metric":["mean", "std", "max"],"Send_period": [mean_send_period, std_send_period, max_send_period]})
        results.to_csv(self.output_file, index=False)
        print("Producer stopped.")
        

class Consumer(TransformBlock):
    def __init__(self, iring, gulp_nframe=1,output_file=None, 
             *args, **kwargs):
        super(Consumer, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.recv_times = []
        self.process_times = []
        self.n_recv = 0
        self.output_file = output_file
        self.first_timestamp = None
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        now = time.monotonic()
        timestamp = idata[0,0,0]
        # print(f"Consumer got data gulp: {idata.shape}, {idata}")
        if self.n_recv < 500:
            self.n_recv += 1
            return None  # warmup phase, don't record times
        else:
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.recv_times.append(now)
            self.process_times.append(now - timestamp)
            self.n_recv += 1
            return None
    def shutdown(self) -> None:
        n_messages = len(self.recv_times)
        print(f"Total messages processed: {n_messages}")
        recv_times_diff = np.diff(self.recv_times)
        mean_recv_period = np.mean(recv_times_diff)*1e6
        std_recv_period = np.std(recv_times_diff)*1e6
        max_recv_period = np.max(recv_times_diff)*1e6

        print(f"Average receive period (us): {mean_recv_period:.4f}")
        print(f"Max receive period (us): {max_recv_period:.4f}, idx: {np.argmax(recv_times_diff)}")
        print(f"Std receive period (us): {std_recv_period:.4f}")

        mean_latency = np.mean(self.process_times)*1e6
        std_latency = np.std(self.process_times)*1e6
        max_latency = np.max(self.process_times)*1e6

        throughput = n_messages / (self.recv_times[-1] - self.first_timestamp)/1e3
        print(f"Throughput (msg/ms): {n_messages}/{(self.recv_times[-1] - self.first_timestamp)*1e3:.4f}ms={throughput:.2f}")

        results = pd.DataFrame({"Metric":["mean", "std", "max"],
                                "Recv_period": [mean_recv_period, std_recv_period, max_recv_period], 
                                "Latency": [mean_latency, std_latency, max_latency], 
                                "Throughput": [throughput, None, None]})

        results.to_csv(self.output_file, index=False)

        print("Consumer stopped.")
        self.shutdown_event.set()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_messages", type=int, default=1000)
    parser.add_argument("--msg_size", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--max_buffer_size", type=int, default=10)  # not used
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output_file", type=str, default="send_times.csv")
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    if os.path.exists(f"results/{args.num_channels}_{args.msg_size}_Consumer.csv") and not args.overwrite:
        return  # Skip if already exists

    data1 = Producer(['random'], n_messages=args.n_messages, n_channels=args.num_channels, msg_size=args.msg_size,output_file=f"results/{args.num_channels}_{args.msg_size}_Producer.csv")

    data2 = Consumer(data1, output_file=f"results/{args.num_channels}_{args.msg_size}_Consumer.csv", gulp_nframe=1)

    pipeline = bf.get_default_pipeline()
    # pipeline.dot_graph().render(filename='benchmark', format='png', cleanup=True)

    pipeline.run()

    pipeline.shutdown()
    print(f"Benchmark complete")

if __name__ == "__main__":
    main()

# async_slide_block.py


import os
import numpy as np

from copy import deepcopy
import time
import bifrost as bf
from meegkit.phase import ECHT
from scipy.signal import iirfilter, lfilter, lfilter_zi, sosfilt, sosfilt_zi, butter

from bifrost.pipeline import TransformBlock, MultiTransformBlock


class SineSourceBlock(bf.pipeline.SourceBlock):
    """Generate a sine wave as a SourceBlock.

    Args:
        sourcenames: list of source names (one entry per sequence)
        sampling_rate: samples per second (Hz)
        frequency: sine frequency (Hz)
        n_samples: optional maximum number of samples to produce (None for infinite)
        gulp_nframe: number of frames per gulp
    """
    def __init__(self, sourcenames, sampling_rate=10000, frequency=10.0,
                 seconds=None, gulp_nframe=1, *args, **kwargs):
        super(SineSourceBlock, self).__init__(sourcenames, gulp_nframe=gulp_nframe, *args, **kwargs)
        self.sampling_rate = float(sampling_rate)
        self.frequency = float(frequency)
        self.dt = 1.0 / self.sampling_rate
        self.n_samples = int(seconds * self.sampling_rate)
        self._emitted = 0
        self.dtype = 'f32'

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
                'dtype':  self.dtype,
                'shape':  [-1, self.gulp_nframe],
                'labels':  ['streamed', 'gulped'],
                'units': [None, None],
                'scales': [[0, 1], [0, 1]]
            },
        }
        return [ohdr]

    def on_data(self, reader, ospans):
        # Fill the output span with sine samples, sleeping between samples
        ospan = ospans[0]
        out = ospan.data
        
        # compute sample
        sample = np.float32(np.sin(2.0 * np.pi * self.frequency * (self._emitted * self.dt)))
        # sample = np.float32((self._emitted * self.dt))

        out[:] = sample
        self._emitted += 1
        # if self._emitted % (self.sampling_rate*1) == 0:
        #     print('sleeping...')
        #     time.sleep(0.1)  # simulate some delay every second
        # Stop if we've emitted enough samples
        if (self._emitted >= self.n_samples):
            print('done emitting samples')
            self.shutdown()
        return [1]
    

class NoiseBlock(TransformBlock):
    def __init__(self, iring, dtype=None, gulp_nframe=1, noise_level=0.1,
             *args, **kwargs):
        super(NoiseBlock, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.dtype  = dtype
        self.noise_level = noise_level
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        out = ospan.data
        # print(f"{self.name} got data gulp: {idata.shape}, {idata}")
        out[:] = idata + np.random.normal(0,self.noise_level,idata.shape)
        return

class FilterBlock(TransformBlock):
    def __init__(self, iring, dtype=None, gulp_nframe=1, fs=None, order=2, freq=None,filter_type='bandstop',
             *args, **kwargs):
        super(FilterBlock, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.dtype  = dtype
        self.order = order
        self.freq = freq  # Keep frequencies in Hz, not normalized
        self.filter_type = filter_type
        self.fs = fs
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)

        self.sos = butter(self.order, self.freq, btype=self.filter_type, output="sos", fs=self.fs)
        # Initialize filter state for filtering along axis=0 with shape (gulp_nframe, 1)
        # zi shape needs to be (n_sections, 2, 1) for axis=0 filtering
        zi_base = sosfilt_zi(self.sos)  # shape: (n_sections, 2)
        self.zi = zi_base[:, :, np.newaxis] * 0  # expand to (n_sections, 2, 1)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        out = ospan.data
        # print(f"{self.name} got data gulp: {idata.shape}, {idata}")
        y, self.zi = sosfilt(self.sos, idata, zi=self.zi, axis=0)
        out[:] = y
        return 


class PrintBlock(TransformBlock):
    def __init__(self, iring, gulp_nframe=1,
             *args, **kwargs):
        super(PrintBlock, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        # print(f"{self.name} got data: {idata.shape}, {idata}")
        return None


class ECHTBlock(TransformBlock):
    def __init__(self, iring, f0,fs,dtype=None, gulp_nframe=1,win_size=1,
             *args, **kwargs):
        super(ECHTBlock, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.dtype  = dtype
        self.win_size = win_size
        self.f0 = f0
        self.fs = fs
    def define_input_overlap_nframe(self, iseq):
        return self.win_size - self._gulp_nframe
    def define_output_nframes(self, input_nframe):
        # return one value per window (mean of the window), default would be n_input_frames=n_output_frames
        return 1
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        filt_BW = self.f0 / 2
        l_freq = self.f0 - filt_BW / 2
        h_freq = self.f0 + filt_BW / 2
        self.echt = ECHT(l_freq, h_freq, self.fs,filt_order=1, calibrate=True, f0=self.f0)
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        # print(f"{self.name} got data gulp: {idata.shape}, {idata[0:11]}")
        ospan.data[:] = np.angle(self.echt.fit_transform(idata))[-1]
        # print(f"{self.name}, output: {odata.shape}, {odata}")
        return 

    
class ECHTWorkerBlock(TransformBlock):
    def __init__(self, iring, f0,fs,dtype=None, gulp_nframe=1,start_offset=0, num_workers=2, win_size=1,
             *args, **kwargs):
        super(ECHTWorkerBlock, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.dtype  = dtype
        self.count = start_offset
        self.num_workers = num_workers
        self.win_size = win_size
        self.f0 = f0
        self.fs = fs
    def define_input_overlap_nframe(self, iseq):
        return self.win_size - self._gulp_nframe
    def define_output_nframes(self, input_nframe):
        # return one value per window (mean of the window), default would be n_input_frames=n_output_frames
        return 1
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        filt_BW = self.f0 / 2
        l_freq = self.f0 - filt_BW / 2
        h_freq = self.f0 + filt_BW / 2
        self.echt = ECHT(l_freq, h_freq, self.fs,filt_order=1)
        return ohdr
    def on_data(self, ispan, ospan):
        if self.count % self.num_workers == 0:
            idata = ispan.data
            odata = ospan.data
            odata[:] = np.angle(self.echt.fit_transform(idata)[-1])
            # print(f"{self.name}, output: {odata.shape}, {odata}")
            self.count += 1
            return 
        else:
            self.count += 1
            return 

class MergeBlock(MultiTransformBlock):
    def __init__(self, irings, dtype=None, gulp_nframe=1,
             *args, **kwargs):
        super(MergeBlock, self).__init__(irings, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        self.dtype  = dtype
        self.num_workers =len(irings)
        self.orings=[self.create_ring(space=irings[0].iring.space)]
    def on_sequence(self, iseqs):
        ihdr = iseqs[0].header
        ohdr = deepcopy(ihdr)
        self.counter = 0
        return [ohdr]
    def on_data(self, ispans, ospans):
        # print(f"Got data spans: {(ispan.data for ispan in ispans)}")
        odata = ospans[0].data
        odata[:] = ispans[self.counter].data
        self.counter = (self.counter+1)%(self.num_workers)

        # print(f"Merged output gulp: {odata.shape}, {odata}")
        return


filename = ['./test_f32.bin']


print("Building pipeline")

fs  = 10000
f0  = 10
win_size = int((1/f0 *2)*fs) # 2 cycles
print(f"win_size: {win_size} samples")

data1 = SineSourceBlock(['sine'], sampling_rate=fs, frequency=10.0, seconds=10, gulp_nframe=1,core=1)
bf.blocks.BinaryFileWriteBlock(data1, gulp_nframe=1000, file_ext='orig.bin')

# data2 = NoiseBlock(data1, dtype='f32', gulp_nframe=win_size//5, noise_level=0, name='noise')
# bf.blocks.BinaryFileWriteBlock(data2, gulp_nframe=1000, file_ext='noisy.bin')

# data3 = FilterBlock(data2, dtype='f32', gulp_nframe=win_size//5, fs=fs,order=3, freq=[9,11], filter_type='bandpass', name='filter',core=2)

window_shift = int(1/f0 *0.05 * fs) # generate output on every sample (f0/fs) or every 5% of cycle
print(f"window_shift: {window_shift} samples")
data4 = ECHTBlock(data1, f0=f0, fs=fs, gulp_nframe=window_shift, win_size=win_size, name='ecHT',core=3)

bf.blocks.BinaryFileWriteBlock(data4, gulp_nframe=1000, file_ext='echt.bin')

print("Running pipeline")
pipeline = bf.get_default_pipeline()
pipeline.dot_graph().render(filename='pipeline_graph_echt', format='png', cleanup=True)

start = time.time()
pipeline.run()

print(f"Needed time: {time.time() - start} seconds")
pipeline.shutdown()

print("All done")
 


# data3 = ECHTWorkerBlock(data1, f0=f0, fs=fs, gulp_nframe=window_shift, start_offset=0, win_size=win_size,num_workers=3,blockname='echt_worker_1',core=1)
# data4 = ECHTWorkerBlock(data1, f0=f0, fs=fs, gulp_nframe=window_shift, start_offset=-1, win_size=win_size,num_workers=3,blockname='echt_worker_2',core=2)
# data5 = ECHTWorkerBlock(data1, f0=f0, fs=fs, gulp_nframe=window_shift, start_offset=-2, win_size=win_size,num_workers=3,blockname='echt_worker_3',core=3)
# data2 = MergeBlock([data3, data4, data5], gulp_nframe=1, core=4)
"""LOGIC
- each worker reads each sliding window (so each time a new sample arrives)
- but only one worker processes it, based on the start_offset and num_workers
- the others just pass and commit a zero to their output buffer
- the outputs are merges together by taking the value from the worker that matches the count 
- next step: dont even step into on_data for the workers that are not supposed to process the window but directly commit zero?
- how to handle if one worker crashes
"""

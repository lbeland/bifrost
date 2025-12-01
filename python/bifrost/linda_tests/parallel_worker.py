# async_slide_block.py


import os
import numpy as np

from copy import deepcopy
import time
import bifrost as bf

from bifrost.pipeline import TransformBlock, MultiTransformBlock




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
        out = ospan.data
        out[:] = idata
        print(f"{self.name} got data gulp: {idata.shape}, {idata}")
        return None


class CustAccumulateBlock(TransformBlock):
    def __init__(self, iring, nframe, dtype=None, gulp_nframe=1, blockname=None,
             *args, **kwargs):
        super(CustAccumulateBlock, self).__init__(iring, gulp_nframe=gulp_nframe, name=blockname,
                                              *args, **kwargs)
        self.nframe = nframe
        self.dtype  = dtype
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        otensor = ohdr['_tensor']
        if 'scales' in otensor:
            frame_axis = otensor['shape'].index(-1)
            otensor['scales'][frame_axis][1] *= self.nframe
        if self.dtype is not None:
            otensor['dtype'] = self.dtype
        self.frame_count = 0
        return ohdr
    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        # print(f"Input_span: {idata}")
        # print(f"Output_span before calc: {odata}")
        odata[:] = idata * 5
        # print(f"{self.name}, output gulp: {odata.shape}, {odata}")
        
        return self._gulp_nframe
    
    
class CustWorkerBlock(TransformBlock):
    def __init__(self, iring, nframe, dtype=None, gulp_nframe=1, blockname=None,start_offset=0, num_workers=2, win_size=1,
             *args, **kwargs):
        super(CustWorkerBlock, self).__init__(iring, gulp_nframe=gulp_nframe, name=blockname,
                                              *args, **kwargs)
        self.nframe = nframe
        self.dtype  = dtype
        self.count = start_offset
        self.num_workers = num_workers
        self.win_size = win_size
    def define_input_overlap_nframe(self, iseq):
        return self.win_size - self._gulp_nframe
    def define_output_nframes(self, input_nframe):
        # return one value per window (mean of the window), default would be n_input_frames=n_output_frames
        return 1
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    def on_data(self, ispan, ospan):
        if self.count % self.num_workers == 0:
            idata = ispan.data
            odata = ospan.data
            odata[:] = np.mean(idata)
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
data1 = bf.blocks.BinaryFileReadBlock(filename, gulp_size=1, gulp_nframe=1, dtype='f32')

data3 = CustWorkerBlock(data1, nframe=1, gulp_nframe=1, start_offset=0, win_size=5,num_workers=3,blockname='worker_1')
data4 = CustWorkerBlock(data1, nframe=1, gulp_nframe=1, start_offset=-1, win_size=5,num_workers=3,blockname='worker_2')
data5 = CustWorkerBlock(data1, nframe=1, gulp_nframe=1, start_offset=-2, win_size=5,num_workers=3,blockname='worker_3')


data5= MergeBlock([data3, data4,data5], gulp_nframe=1)      #has to be 1 in current implementation
monitor = PrintBlock(data5, gulp_nframe=5)

print("Running pipeline")
pipeline = bf.get_default_pipeline()
pipeline.dot_graph().render(filename='pipeline_graph_workers', format='png', cleanup=True)

start = time.time()
pipeline.run()

print(f"Needed time: {time.time() - start} seconds")
pipeline.shutdown()

print("All done")
 
"""LOGIC
- each worker reads each sliding window (so each time a new sample arrives)
- but only one worker processes it, based on the start_offset and num_workers
- the others just pass and commit a zero to their output buffer
- the outputs are merges together by taking the value from the worker that matches the count 
- next step: dont even step into on_data for the workers that are not supposed to process the window but directly commit zero?
- how to handle if one worker crashes
"""

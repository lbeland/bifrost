from bifrost.pipeline import Pipeline,TransformBlock, SourceBlock, MultiTransformBlock
import bifrost as bf
import numpy as np
from copy import deepcopy

class PrintBlock(TransformBlock):
    def __init__(self, iring, gulp_nframe=1,
             *args, **kwargs):
        super(PrintBlock, self).__init__(iring, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    # def define_input_overlap_nframe(self, iseqs):
    #     """Return no. input frames that should overlap between successive spans
    #     for each input sequence.
    #     """
    #     return 4
    def on_data(self, ispan, ospan):
        idata = ispan.data
        out = ospan.data
        out[:] = idata
        print(f"{self.name} got data gulp: {idata.shape}, {idata}")
        return len(idata)


class MergeBlock(MultiTransformBlock):
    def __init__(self, irings, dtype=None, gulp_nframe=1,
             *args, **kwargs):
        super(MergeBlock, self).__init__(irings, gulp_nframe=gulp_nframe,
                                              *args, **kwargs)
        # self.orings = [self.create_ring(space=self.irings[0].space)]
        self.dtype  = dtype
    def on_sequence(self, iseqs):
        ihdr = iseqs[0].header
        ohdr = deepcopy(ihdr)
        print(ohdr['_tensor']['shape'])
        ohdr['_tensor']['shape'] = [-1,2]
        return [ohdr, ohdr]


    def on_data(self, ispans, ospan):
        """
        ispans: list of input spans
        ospan: output span
        """
        # Allocate output array
        out = ospan[0].data  # shape (2, 1)
        print(f"MergeBlock got input spans: {[ispan.data for ispan in ispans]}")

        # Each input has shape (1,) since gulp=1
        out[0,0] = ispans[0].data[0]  # first input
        out[0,1] = ispans[1].data[0]  # second input

        if len(ispans[0].data[0]) < 3:
            return [1]
        return [1]  # commit the frame

filename = ['/home/linda/bifrost/test_f32.bin']

# Example: two sources
data1 = bf.blocks.BinaryFileReadBlock(filename, gulp_size=1, gulp_nframe=1, dtype='f32')
data2 = bf.blocks.BinaryFileReadBlock(filename, gulp_size=1, gulp_nframe=1, dtype='f32')

merge1 = MergeBlock(irings=[data1, data2], gulp_nframe=1)

monitor = PrintBlock(merge1, gulp_nframe=1)

# monitor2 = PrintBlock(monitor, gulp_nframe=1)


pipeline = bf.get_default_pipeline()
pipeline.dot_graph().render(filename='pipeline_graph_merge', format='png', cleanup=True)
pipeline.run()
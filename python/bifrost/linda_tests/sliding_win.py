from bifrost.pipeline import Pipeline,TransformBlock, SourceBlock, MultiTransformBlock
import bifrost as bf
from copy import deepcopy
import matplotlib.pyplot as plt

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
    #     plt.figure()
    #     plt.plot(ispan.data.flatten())
    #     plt.show(block=False)
        idata = ispan.data
        print(f"PrintBlock got data gulp: {idata.shape}, {idata}")
        return None


class SlideBlock(TransformBlock):
    def __init__(self, iring, dtype=None, gulp_nframe=1, win_size=3, blockname=None,
             *args, **kwargs):
        super(SlideBlock, self).__init__(iring, gulp_nframe=gulp_nframe, name=blockname,
                                              *args, **kwargs)
        self.dtype  = dtype
        self.win_size = win_size
    def on_sequence(self, iseq):
        ihdr = iseq.header
        ohdr = deepcopy(ihdr)
        return ohdr
    def define_input_overlap_nframe(self, iseq):
        return self.win_size - self._gulp_nframe
    def on_data(self, ispan, ospan):
        idata = ispan.data
        out = ospan.data
        out[:] = idata
        # print(f"{self.name} got data gulp: {idata.shape}, {idata}")
        return len(idata)


filename = ['/home/linda/bifrost/test_f32.bin']

# Example: two sources
data1 = bf.blocks.BinaryFileReadBlock(filename, gulp_size=1, gulp_nframe=1, dtype='f32')
data2 = SlideBlock(data1, dtype='f32', gulp_nframe=3, win_size=5)


monitor = PrintBlock(data2, gulp_nframe=5)


pipeline = bf.get_default_pipeline()
# pipeline.dot_graph().render(filename='pipeline_graph_slide', format='png', cleanup=True)
pipeline.run()
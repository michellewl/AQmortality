class S5P_Filename():
    def __init__(self, filename):
        self.filename = filename
        self.mission = filename[0:3]
        self.stream = filename[4:8]
        self.product = filename[9:19]
        self.granule_start = filename[20:35]
        self.granule_end = filename[36:51]
        self.orbit = filename[52:57]
        self.collection = filename[58:60]
        self.processor_version = filename[61:67]
        self.processing_time = filename[68:83]
        self.extension = filename[83:86]
from collections import OrderedDict
import sys

def unique_index(L,e):
    id = None
    iCnt = 0
    for l in L:
        if str(l) == str(e) :
            id = iCnt
        iCnt = iCnt + 1
    if id == None:
        raise ValueError
    return id
    
class Msnhnet:
    Export = False
    Debug = False
    msnh_layer_cnt = -1
    def __init__(self):
        self.inAddr = ""
        self.net = ""
        self.index = 0
        self.names = []
        self.indexes = []

    def setNameAndIdx(self, name, ids):
        self.names.append(name)
        self.indexes.append(ids)

    def getIndexFromName(self,name):
        ids = self.indexes[unique_index(self.names,name)]
        return ids

    def getLastVal(self):
        return self.indexes[-1]

    def getLastKey(self):
        return self.names[-1]

    def checkInput(self, inAddr,fun):

        if self.index == 0:
            return

        if str(inAddr._cdata) != self.getLastKey():
            try:
                ID = self.getIndexFromName(str(inAddr._cdata))
                self.buildRoute(str(inAddr._cdata),str(ID),False,True)
            except:
                 raise NotImplementedError("last op is not supported " + fun + str(inAddr._cdata))
            

    def buildConfig(self, inAddr, shape):
        print("========== Layer config       : ==========\n")
        self.inAddr = inAddr
        self.net = self.net + "config:\n"
        self.net = self.net + "  batch: " + str(int(shape[0])) + "\n"
        self.net = self.net + "  channels: " + str(int(shape[1])) + "\n"
        self.net = self.net + "  height: " + str(int(shape[2])) + "\n"
        self.net = self.net + "  width: " + str(int(shape[3])) + "\n"

    def buildEmpty(self, name):
        self.setNameAndIdx(name,self.index)
        self.index = self.index + 1

 
    def buildConv2d(self, name, filters, kSizeX, kSizeY, paddingX, paddingY, strideX, strideY, dilationX, dilationY, groups, useBias, inshape, outshape):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer conv2d       : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        self.net = self.net + "conv:\n"
        self.net = self.net + "  filters: " + str(int(filters)) + "\n"
        self.net = self.net + "  kSizeX: " + str(int(kSizeX)) + "\n"
        self.net = self.net + "  kSizeY: " + str(int(kSizeY)) + "\n"
        self.net = self.net + "  paddingX: " + str(int(paddingX)) + "\n"
        self.net = self.net + "  paddingY: " + str(int(paddingY)) + "\n"
        self.net = self.net + "  strideX: " + str(int(strideX)) + "\n"
        self.net = self.net + "  strideY: " + str(int(strideY)) + "\n"
        self.net = self.net + "  dilationX: " + str(int(dilationX)) + "\n"
        self.net = self.net + "  dilationY: " + str(int(dilationY)) + "\n"
        self.net = self.net + "  groups: " + str(int(groups)) + "\n"
        self.net = self.net + "  useBias: " + str(int(useBias)) + "\n"

    def buildActivation(self, name, activation, inshape, outshape, params=None):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer activation   : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        self.net = self.net + "act:\n"
        
        # todo 
        if activation == "selu":
            self.net = self.net + "  activation: selu\n"
            return
        if activation == "elu":
            self.net = self.net + "  activation: elu\n"
            return
        if activation == "relu":
            self.net = self.net + "  activation: relu\n"
            return
        if activation == "relu6":
            self.net = self.net + "  activation: relu6\n"    
            return
        if activation == "sigmoid":
            self.net = self.net + "  activation: logistic\n" 
            return
        if activation == "leaky":
            self.net = self.net + "  activation: leaky,"+str(params)+"\n"
            return
        if activation == "tanh":
            self.net = self.net + "  activation: tanh\n" 
            return
        if activation == "logistic":
            self.net = self.net + "  activation: logistic\n" 
            return
        if activation == "softplus":
            self.net = self.net + "  activation: softplus,"+str(params)+"\n"
            return
        if activation == "linear":
            self.net = self.net + "  activation: none\n"
            return
        if activation == "hardswish":
            self.net = self.net + "  activation: hardswish\n"
            return
        if activation == "silu":
            self.net = self.net + "  activation: swish\n"
            return
        raise NotImplementedError("unknown actiavtion : "+activation)
        

    def buildSoftmax(self, name, inshape, outshape):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer softmax      : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        self.net = self.net + "softmax:\n"
        self.net = self.net + "  groups: 1\n"
    
    def buildBatchNorm(self, name , inshape, outshape , eps=0.00001):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer batchnorm    : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        self.net = self.net + "batchnorm:\n"
        self.net = self.net + "  activation: none\n"
        self.net = self.net + "  eps: "+str(float(eps))+"\n"

    def buildGlobalAvgPooling(self, name, inshape, outshape):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer globalAvgpool: ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        self.net = self.net + "globalavgpool:\n  "
        
    def buildPooling(self, name, type, kSizeX, kSizeY, strideX, strideY, paddingX, paddingY, ceilMode, inshape, outshape):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer pooling      : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        if type == "MAX" :
            self.net = self.net + "maxpool:\n"
        else:
            self.net = self.net + "localavgpool:\n"

        self.net = self.net + "  kSizeX: " + str(int(kSizeX)) + "\n"
        self.net = self.net + "  kSizeY: " + str(int(kSizeY)) + "\n"
        self.net = self.net + "  paddingX: " + str(int(paddingX)) + "\n"
        self.net = self.net + "  paddingY: " + str(int(paddingY)) + "\n"
        self.net = self.net + "  strideX: " + str(int(strideX)) + "\n"
        self.net = self.net + "  strideY: " + str(int(strideY)) + "\n"
        self.net = self.net + "  ceilMode: " + str(int(ceilMode)) + "\n"

    def buildConnect(self, name, output, useBias, inshape, outshape):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer connect      : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index)+" "+ str(inshape) +"  -->  " + str(outshape) +"\n"
        self.index = self.index + 1
        self.net = self.net + "connect:\n"
        self.net = self.net + "  output: " + str(int(output)) + "\n"
        self.net = self.net + "  useBias: " + str(int(useBias)) + "\n"

    def buildUpsample2D(self, name, strideX, strideY, scaleX, scaleY, type, alignCorners):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer upsamle2D    : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "upsample:\n"
        self.net = self.net + "  type: " + type + "\n"
        self.net = self.net + "  strideX: " + str(int(strideX)) + "\n"
        self.net = self.net + "  strideY: " + str(int(strideY)) + "\n"
        self.net = self.net + "  scaleX: " + str(float(scaleX)) + "\n"
        self.net = self.net + "  scaleY: " + str(float(scaleY)) + "\n"
        self.net = self.net + "  alignCorners: " + str(int(alignCorners)) + "\n"

#注意，自定义时非cat, 自行处理输入name, 不要给c_data
    def buildRoute(self, name, layers, addModel, isCheckBuild = False):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer route        : ==========",Msnhnet.msnh_layer_cnt,"\n")
        if isCheckBuild:
            self.setNameAndIdx("route",self.index)
        else:
            self.setNameAndIdx(name,self.index)

        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "route:\n"
        self.net = self.net + "  layers: " + layers + "\n"
        self.net = self.net + "  addModel: " + str(int(addModel)) + "\n"

    def buildPadding(self, name, top, down, left, right):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer padding      : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "padding:\n"
        self.net = self.net + "  top: " + str(int(top)) + "\n"
        self.net = self.net + "  down: " + str(int(down)) + "\n"
        self.net = self.net + "  left: " + str(int(left)) + "\n"
        self.net = self.net + "  right: " + str(int(right)) + "\n"
        self.net = self.net + "  paddingVal: 0\n"
    
    def buildVariableOp(self, name, layers, type, constVal=0):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer varop        : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "varop:\n"
        if layers!= "" :
            self.net = self.net + "  layers: " + layers + "\n"
        self.net = self.net + "  type: "   + type + "\n"
        self.net = self.net + "  constVal: "   + str(float(constVal)) + "\n"
        
    def buildPermute(self, name, dim0, dim1, dim2):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer permute      : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "permute:\n"
        self.net = self.net + "  dim0: " + str(int(dim0-1)) + "\n"
        self.net = self.net + "  dim1: " + str(int(dim1-1)) + "\n"
        self.net = self.net + "  dim2: " + str(int(dim2-1)) + "\n"

    def buildReduction(self, name, type, axis):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer reduction    : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "reduction:\n"
        self.net = self.net + "  type: " + type + "\n"
        self.net = self.net + "  axis: " + str(int(axis-1)) + "\n"

    def buildView(self, name, dim0, dim1, dim2):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer view         : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "view:\n"
        self.net = self.net + "  dim0: " + str(int(dim0)) + "\n"
        self.net = self.net + "  dim1: " + str(int(dim1)) + "\n"
        self.net = self.net + "  dim2: " + str(int(dim2)) + "\n"

    def buildSlice(self, name, start0, step0, start1, step1, start2, step2):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer slice        : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "slice:\n"
        self.net = self.net + "  start0: " + str(int(start0)) + "\n"
        self.net = self.net + "  step0: " + str(int(step0)) + "\n"
        self.net = self.net + "  start1: " + str(int(start1)) + "\n"
        self.net = self.net + "  step1: " + str(int(step1)) + "\n"
        self.net = self.net + "  start2: " + str(int(start2)) + "\n"
        self.net = self.net + "  step2: " + str(int(step2)) + "\n"

    def buildYolo(self,name, anchors, classNum, yoloType="yolov3"):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer yolo         : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "yolo:\n"
        self.net = self.net + "  anchors: " + anchors + "\n"
        self.net = self.net + "  classNum: " + str(int(classNum)) + "\n"   
        self.net = self.net + "  yoloType: " + yoloType +"\n"  

    def buildYoloOut(self,name, yoloLayers ,yoloType="yolov3"):
        Msnhnet.msnh_layer_cnt = Msnhnet.msnh_layer_cnt+1
        print("========== Layer yoloout      : ==========",Msnhnet.msnh_layer_cnt,"\n")
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "yoloout:\n"
        self.net = self.net + "  layers: " + yoloLayers + "\n" 
        self.net = self.net + "  confThresh: 0.5\n"
        self.net = self.net + "  nmsThresh: 0.5\n"     
        self.net = self.net + "  useSoftNms: 0\n"  
        self.net = self.net + "  yoloType: " + yoloType +"\n"  
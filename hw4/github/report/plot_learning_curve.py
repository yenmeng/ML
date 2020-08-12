import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

""" train """
loss = [0.4563320640393775, 0.4052779364746913, 0.38523687503853177, 0.3676745182693195, 0.34886440918130296, 0.32791897857947744, 0.30323701894319316, 0.27624316534462495, 0.24847344398615073, 0.2228509571497291]
acc = [78.05834518307856, 81.58382954141487, 82.62808833985069, 83.54403661571276, 84.63773107003199, 85.73975737646641, 87.0450808745112, 88.45371933878423, 89.8268085673658, 91.06325542125843]
acc = [x/100 for x in acc]
epoch = np.arange(1,11,1)

plt.figure(1)
plt.plot(epoch,loss,'b--',epoch,acc,'r--')
plt.legend(['loss','accuracy'])
plt.title('learning curve(train)')
plt.xticks(np.arange(min(epoch), max(epoch)+1, 1))
plt.xlabel('epoch')
plt.savefig('./train.png')



""" validation """
loss_v = [0.44483, 0.39611, 0.40636, 0.39078, 0.38860, 0.40012, 0.40548, 0.43070, 0.48593, 0.51440]
acc_v = [79.403, 81.984, 81.440, 82.059, 82.313, 82.473, 81.864, 81.490, 80.816, 80.791]
acc_v = [x/100 for x in acc_v]
plt.figure(2)
plt.plot(epoch,loss_v,'b--',epoch,acc_v,'r--')
plt.legend(['loss','accuracy'])
plt.title('learning curve(validation)')
plt.xticks(np.arange(min(epoch), max(epoch)+1, 1))
plt.xlabel('epoch')
plt.savefig('./valid.png')

""" loss """
plt.figure(3)
plt.plot(epoch,loss,epoch,loss_v)
plt.legend(['train','validation'])
plt.title('LOSS')
plt.xticks(np.arange(min(epoch), max(epoch)+1, 1))
plt.xlabel('epoch')
plt.savefig('./loss.png')

""" accuracy """
plt.figure(4)
plt.plot(epoch,acc,epoch,acc_v)
plt.legend(['train','validation'])
plt.title('ACCURACY')
plt.xticks(np.arange(min(epoch), max(epoch)+1, 1))
plt.xlabel('epoch')
plt.savefig('./accuracy.png')



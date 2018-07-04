import matplotlib.pyplot as plt
import os
#plt.style.use('ggplot')
#plt.style.use('presentation')
if __name__ == '__main__':
    fn_list = ['vgg11_bn', 'depth_fire', 'mobile', 'fire']
    ln_list = ['VGG11_bn', 'Depthwise fire', 'Depthwise & pointwise', 'Fire']

    x_all = []*len(fn_list)
    y_all = []*len(fn_list)
    for i in range(len(fn_list)):
        f = open(os.path.join('q_result', fn_list[i]), 'r')
        x, y = [], []
        for line in f:
            num_bits = int(line.split(',')[0])
            acc = float(line.split(',')[1])             
            size = float(line.split(',')[2])

            x.append(num_bits)
            y.append(acc)

        x_all.append(x)
        y_all.append(y)
        
    for i in range(len(x_all)):
        plt.plot(x_all[i], y_all[i], label = ln_list[i])
        plt.scatter(x_all[i], y_all[i])
        plt.xlabel('Number of bits')
        plt.ylabel('Accuracy (%)')


    plt.hlines(70, xmin = 2, xmax = 32, color = 'r', label = 'Strong baseline', linestyles = 'dashed')
    plt.xticks(range(1,33))
    plt.yticks(range(0,86,5))
    plt.legend(loc = 4, prop={'size':20})

    plt.show()
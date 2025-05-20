import torch
import numpy as np
import scipy.stats as stats
import math



class fitpartition:
    sum_p2 = 0
    sum_p1 = 0
    alpha = 0
    N = []
    delta = 0
    loc = 0
    new = [0]
    ed = math.e
    new = 0

    def __init__(self, alpha, delta, mark_info, B):
        self.alpha = alpha
        self.delta = delta
        self.new = [0] * B
        self.mark_info = mark_info
        self.N, self.loc = [0] * B, [0] * B
        self.sum_p2, self.sum_p1 = np.zeros(B), np.zeros(B)
        self.ed = pow(math.e, delta)
        self.zpow = pow(stats.norm.ppf(1-self.alpha),2)
        self.sigma1 = [[] for _ in range(B)]
        self.bitcount = [0]*B
        self.seg = [[] for _ in range(B)]
    def cal_partition(self, l_t_input, index, gli, gls,in_green_list):
        self.N[index] = self.N[index] + 1
        l_t = torch.softmax(l_t_input, dim=-1)
        z_1 = l_t * (gli < gls)
        z_2 = l_t * (gli >= gls)
        
        sigma_1 = torch.sum(z_1).item()
        self.sigma1[index].append(sigma_1)
        sigma_2 = torch.sum(z_2).item()
        alpha2 = pow(self.alpha,2)

        # l = np.exp([in_green_list[index].item(),self.N[index]-in_green_list[index].item()-1])
        # l = l / l.sum()
        # p =self.ed * (l[0]*sigma_1/(self.ed * sigma_1+sigma_2) + l[1]*sigma_2/(self.ed * sigma_2+sigma_1))
        # # print(f"({in_green_list[index]})",end=" ")
        
        p = [self.ed * ((in_green_list[index]+ 0.5)*s/(self.ed * s+1-s) + (self.N[index]-in_green_list[index]-0.5)*(1-s)/(self.ed * (1-s)+s))/(self.N[index]) for s in self.sigma1[index]]
        self.sum_p2[index] = sum(x**2 for x in p)
        self.sum_p1[index] = sum(p)
        
        # p =self.ed * ((in_green_list[index]+ 1)*sigma_1/(self.ed * sigma_1+sigma_2) + (self.N[index]-in_green_list[index])*sigma_2/(self.ed * sigma_2+sigma_1))/(self.N[index]+1)
        # self.sum_p2[index] += p*p
        # self.sum_p1[index] += p



        left =  self.zpow
        right = pow((self.sum_p1[index] - 0.5*self.N[index]), 2)/(self.sum_p1[index] - self.sum_p2[index])        
        if left <= right:
            self.sum_p2[index], self.sum_p1[index] = 0, 0
            self.sigma1[index] = []
            self.N[index] = 0
            self.new[index] = right - left
            return 1
        else:
            return 0

    def markbit_by_partition(self, l_t_input_batch, gli_batch, gls,in_green_list,_):
        
        
        for i in range(len(in_green_list)):
            if self.new[i]:
                self.seg[i].append(_)
                if len(in_green_list) == 1:
                    print("|||")
                self.new[i] = 0
                in_green_list[i] = 0

                
        self.loc = [loc % len(self.mark_info) for loc in self.loc]
        bit_batch = [self.mark_info[i] for i in self.loc]
        changebit = [self.cal_partition(
            l_t_input, index, gli, gls,in_green_list) for l_t_input, index, gli in zip(l_t_input_batch, range(l_t_input_batch.shape[0]), gli_batch)]
        self.bitcount = [count + b for count, b in zip(self.bitcount, changebit)]
        self.loc = [loc + b for loc, b in zip(self.loc, changebit)]
        return bit_batch,self.N,self.bitcount,self.seg

    def markloc_by_partition(self, l_t_input_batch, gli_batch, gls,in_green_list):
        change =  [self.cal_partition(
            l_t_input, index, gli, gls,in_green_list) for l_t_input, index, gli in zip(l_t_input_batch, range(l_t_input_batch.shape[0]), gli_batch)]
        
        return change



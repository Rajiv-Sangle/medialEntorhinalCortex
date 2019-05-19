
# coding: utf-8

# In[2]:


from subprocess import call
import numpy as np
import sys
import time as kt

bench_start = kt.time()

nns = [12]
times = [1000.0]
rep = 1

bench = np.zeros((len(nns),len(times),rep))
eps= 0.01

for ni,n in enumerate(nns):
    for nj,t in enumerate(times):
        for r in range(rep):
            
            print("\n\n!!!!! Starting Simulation !!!!!\n\n")
            print("Number of Neurons:",n,"Length of Simulation:",t,"Replicate Number:",r)
            start = kt.time()
            call(['python3','gen_input.py',str(n),str(t),str(eps)])
            
            time = np.split(np.arange(0,t,eps),int(t/100))
            
            for nt,i in enumerate(time):
                if nt>0:
                    time[nt] = np.append(i[0]-0.01,i)
            
            np.save("time",time,allow_pickle=True)

            for i in range(int(t/100)):
                call(['python3','run.py',str(i),str(n),str(eps),str(int(t/100))])

            bench[ni,nj,r]= kt.time()-start
            call(['rm','*.npy'])
            np.save("bench",bench)

bench_end = kt.time()

print("Benchmark ran for",bench_end-bench_start,"seconds")

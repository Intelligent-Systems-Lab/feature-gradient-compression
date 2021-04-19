import sys,os
import subprocess
import time
import glob

def run_capture(File="network"):
    proc = subprocess.Popen('tshark -i bcfl -b filesize:102400 -w {}'.format(File), shell=True, stdout=subprocess.PIPE)
    return proc

def run_formatter(File):
    save = "{}/{}.csv".format(os.path.dirname(File), '_'.join(File.split("/")[-1].split("_")[:2]))
    # tshark -r network_12_56_58.pcap -T fields -e frame.time_relative -e ip.src -e ip.dst -e ip.proto -e tcp.srcport -e tcp.dstport -e frame.len -E header=n -E separator=, -E quote=n -E occurrence=f
    proc = subprocess.Popen('tshark -r {} -T fields -e frame.time_relative -e ip.src -e ip.dst -e ip.proto -e tcp.srcport -e tcp.dstport -e frame.len -E header=n -E separator=, -E quote=n -E occurrence=f > {}'.format(File, save), shell=True, stdout=subprocess.PIPE)
    proc.wait()
    _ = subprocess.Popen('rm {}'.format(File), shell=True, stdout=subprocess.PIPE)

if __name__ == "__main__":
    p_capture = run_capture(sys.argv[1])
    print("PID : {}".format(p_capture.pid))
    time.sleep(5)

    path = os.path.dirname(sys.argv[1])

    while not os.path.isfile("/root/save/capture_down"):
        time.sleep(10)

        l = glob.glob(path+"/network_*.pcap")
        nl = [int(i.split("/")[-1].split("_")[1]) for i in l]
        # print(nl)
        if len(nl):
            nl.remove(max(nl))
        
        if len(nl):
            f = [i for i in l if int(i.split("/")[-1].split("_")[1])==nl[0]][0]
            if not os.path.isfile(f):
                continue
            run_formatter(f)

    p_capture.kill()

    time.sleep(5)

    l = glob.glob(path+"/network_*.pcap")

    for j in l:
        if not os.path.isfile(j):
            continue
        run_formatter(j)
    
    open("/root/save/convert_down", 'a').close()




    
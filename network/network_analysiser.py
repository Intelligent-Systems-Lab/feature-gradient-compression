import sys
# from scapy.utils import RawPcapReader
import matplotlib.pyplot as plt
import glob
import copy
import pandas as pd
import os
# file_name = "/Users/tonyguo/Desktop/test4.pcap"
# file_name = sys.argv[1]


def analysis_pcap(file_name):
    pcap = RawPcapReader(file_name)
    packet_pcap = pcap.read_all()
    pkt_data = []

    time_start = (packet_pcap[0][1].tshigh << 32) | packet_pcap[0][1].tslow
    time_record = [time_start]
    flow = []
    flows = []
    flow.append(packet_pcap[0][1].wirelen)
    flows.append(packet_pcap[0][1].wirelen)
    sum_value = packet_pcap[0][1].wirelen
    for i in packet_pcap[1:]:
        pkt_data.append(copy.deepcopy(i[0]))
        flow.append(i[1].wirelen)
        sum_value += i[1].wirelen
        flows.append(copy.deepcopy(sum_value))
        time_record.append(copy.deepcopy((i[1].tshigh << 32) | i[1].tslow))
    
    time_record = [(t - time_start)/10 ** 9 for t in time_record]
    print("capture {} packets, {} bytes, {} mb".format(len(packet_pcap), flows[-1], flows[-1] / 10 ** 6))
    # pkt_data = pkt_data[:-5]
    # time_record = time_record[:-5]
    # flow = flow[:-5]
    # flows = flows[:-5]

    return pkt_data, time_record, flow, flows


def analysis_csv(files):
    data = None
    files.sort(key=lambda x:int(x.split("/")[-1].split("_")[1].replace(".csv", "")))
    for i in files:
        r = pd.read_csv(i, header=None)
        r.columns = ["time", "src", "dst", "proto", "srcport", "dstport", "frame"]
        r = r.fillna(0)
               
        if data is None: 
            data = copy.deepcopy(r)
        else:
            r["time"] = r["time"] + data["time"].to_list()[-1]
            data = data.append(copy.deepcopy(r), ignore_index=True)
    # return data

    pkt_data = []
    time_record = data["time"].to_list()
    flow = data["frame"].to_list()
    flows = []
    flows.append(flow[0])
    for i in range(1, len(flow)):
        v = flows[-1] + flow[i]
        flows.append(v)

    return pkt_data, time_record, flow, flows


def plt_save(data, time_, save_file, info=""):
    data_ = copy.deepcopy(data)
    plt.title("traffic(MB)-time(s){}".format(info))
    plt.grid(True)
    plt.ylabel("MB")
    plt.xlabel("Time(s)")
    plt.plot(data, time_, color='red')
    plt.savefig(save_file)


if __name__ == "__main__":
    p = glob.glob(sys.argv[1]+"/network_*.csv")
    # _, time_record, flow, flows = analysis_pcap(p)

    _, time_record, flow, flows = analysis_csv(p)
    # print(time_record)
    # print(max(time_record), max(flow))
    #print(flows)
    flows = [i/10**6 for i in flows]
    # flow = [i/10**6 for i in flow]
    # flow_ = []
    # flow_.append(flows[0])
    # for j in range(1,len(flows)):
    #     flow_.append(flows[j]-flows[j-1])
    # flow_ = [flows[j]-flows[j-1] for j in range(1,len(flows))]
    plt_save(time_record, flows, sys.argv[2], "   Max: {}".format(int(flows[-1])))
    # plt_save(time_record, flow_, sys.argv[3])

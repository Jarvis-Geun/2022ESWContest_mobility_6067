import matplotlib.pyplot as plt

def convert_ppg(old_path, new_path):
    img = open(old_path, "r")
    lines = img.readlines()
    signal = []

    f = open(new_path, "w")
    for line in lines:
        line = line.split("'")[1]
        line = line.split('\\')[0]
        # print(line)
        if len(line) != 10:
            line = '-1'
        f.write(line + '\n')
    
    img.close()
    f.close()
    return signal

def visulize_ppg(new_path):
    with open(new_path) as f:  
        ppg_list = f.read().splitlines()

    for i, ppg in enumerate(ppg_list):
        ppg_list[i] = float(ppg_list[i])
    plt.figure(figsize=(50, 1), dpi=500)
    plt.plot(ppg_list)
    # plt.plot(PPG_peaks_fil_sq, ppg[PPG_peaks_fil], 'x')
    # plt.title('subject'+str(Subject_num)+'-'+str(Phase_num))
    plt.savefig("new_ppg.png")

if __name__ == "__main__":
    old_path = "ppg_data.txt"
    new_path = "new_ppg_data.txt"
    convert_ppg(old_path, new_path)
    visulize_ppg(new_path)
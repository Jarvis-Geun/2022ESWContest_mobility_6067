from datetime import datetime
score_list = []
filename = "./LabelData.txt"

def SOFI():

    global score_list
    SOFI_score = []
    print(" === SOIF ===\n")
    print("Physical Exertion: ")
    SOFI_score.extend(list(map(int, input().split())))

    print("Physical Discomfort: ")
    SOFI_score.extend(list(map(int, input().split())))

    print("Lack of Motivation: ")
    SOFI_score.extend(list(map(int, input().split())))

    print("Sleepniess: ")
    SOFI_score.extend(list(map(int, input().split())))

    print("Lack of Energy: ")
    SOFI_score.extend(list(map(int, input().split())))

    MinMaxScalingSOFI = round((((sum(SOFI_score)) / (10*len(SOFI_score)))*10), 6)
    score_list.append(MinMaxScalingSOFI)

    return


def FAS():
    global score_list
    FAS_score = []
    print(" === FAS === \n ")
    print("Input 10 Answers of FAS:  ")
    FAS_score.extend(list(map(int, input().split())))

    MinMaxScalingFAS = round((((sum(FAS_score) - 9) / 36))*10, 6)
    score_list.append(MinMaxScalingFAS)

    return

def VAS():
    global score_list
    VAS_score = []
    print(" === VAS === \n ")
    print("Input 17 Answers of VAS:  ")
    VAS_score.extend(list(map(int, input().split())))

    MinMaxScalingVAS = round((((sum(VAS_score)) / (10*len(VAS_score)))*10), 6)
    score_list.append(MinMaxScalingVAS)

    return

def FatigueScore():
    global filename, score_list

    FatigueScore = round((sum(score_list)), 6)
    with open(filename, 'a') as f:
        f.write(f'{FatigueScore}, {datetime.now().timestamp()}\n')

    return

if __name__ == "__main__":

    while True:
        print("What questionnaire do you want?\n")
        print("CASE: FAS, SOFI, VAS, if you want to end it: END\n")
        print("Write hear: ")

        State = input()
        if State == "FAS":
            FAS()
        elif State == "SOFI":
            SOFI()
        elif State == "VAS":
            VAS()
        elif State == "END":
            if len(score_list) >= 1:
                FatigueScore()
            else:
                print("None Score")
            break
        else:
            print("check Spelling(Input CASE: FAS, SOFI, VAS, END)\n")
            continue

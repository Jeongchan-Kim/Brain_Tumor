import torch

def test(test_dl, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            inputs, targets = data
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            # one-hot encoding 이므로
            correct += (predicted == targets.max(dim=1)[1]).sum().item()
    print(f"total: {total}, correct: {correct}")        
    print(f"Acc: {100 * correct // total} %")

    '''
    # 클래스 별 정확도 계산
    correct_class = {classname: 0 for classname in classes}
    total_class = {classname: 0 for classname in classes}
            for label, prediction in zip(targets, predicted):
                if label == prediction:
                    correct_class[classes[label]] += 1
                total_class[classes[label]] += 1

    for classname, correct_count in correct_class.items():
        if total_class[classname] == 0:
            continue
        accuracy = 100 * float(correct_count) / total_class[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    '''
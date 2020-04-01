import json

with open("train.jsonl",'r') as load_f:
    print('load sucess')
    json_list = list(load_f)
    print(len(json_list))
    print(json_list[0:1])
    #print(json_list[0])
    result = json.loads(json_list[0])
    print(type(result))
    print(type(result['summary']))
    print(result['summary'][0:10])
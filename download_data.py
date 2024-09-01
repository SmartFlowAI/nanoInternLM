import openxlab


openxlab.login(ak='', sk='') # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK

from openxlab.dataset import info
# info(dataset_repo='OpenDataLab/ShareGPT91K') #数据集信息查看

# from openxlab.dataset import get
# get(dataset_repo='OpenDataLab/ShareGPT91K', target_path='/root/code/nanointernlm/data/ShareGPT91K') # 数据集下载

# from openxlab.dataset import download
# download(dataset_repo='OpenDataLab/ShareGPT91K',source_path='/README.md', target_path='/root/code/nanointernlm/data/ShareGPT91K') #数据集文件下载

info(dataset_repo='suntianxiang/moss-003-sft-data') #数据集信息查看

from openxlab.dataset import get
get(dataset_repo='suntianxiang/moss-003-sft-data', target_path='/root/code/nanointernlm/data/moss-003-sft-data') # 数据集下载


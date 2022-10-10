import random
import time
import requests
from fake_useragent import UserAgent
from retrying import retry

#功能：github api 爬虫files（带有身份验证的）
requests.packages.urllib3.disable_warnings()
authorization = [
    "1ab37fd0181c95812b06db1fb7d92157e8973c25",
    "4467f0affcf017d423a319161f095d5d242fa581",
    "ff606734988aefe6964fa63587514015e7314467",
    "648808ae3cd9499fb578805e72ca7ad31efeabab",
    "ff68dbb9028c6c631ca44d275f095a3ffcbe0910",
    "cc332fbad3bb4ad90daa13498d609988a1c60882",
    "e8b5b8b51746bc7f448463bf6c3f0d38ff062235",
    "90529a59a469effbfb02db47a6ea7ec0906a8334",
    "953ec97bb6160b4529324c58a20ec8970a482843",
    "cf1bde01e346fe9f8c88d826c09a51f63dc96e50",
    "bdcf5019e7cacc32919d2053c7187aae3a672b02",
    "f7a6a4bb0e279d104a7d2e3238547bf7e2cc9e60"
    "9123094a2788563a25c9f157967ddf15692be897",
    "b00f61e174317991163fcc970ac97ae974644446",
    "4e88f4c4f4551a9f4a66ac0e72055175c78ddbd9",
    "5042c0c93ac2f28397affb8c126b3350161c9254",
    "7783c540e064248917122b50294ac756dac3a435",
    "5d5aea26057381348ef688a2fb18ad407588fd9a",
    "623b775906882c74d5d280078271e1f97fc9f1e1"
    ]

def changetoken():
    newid = random.choice (authorization)
    return "token " + newid

def changeagent():
    ua = UserAgent()
    return  ua.random

def change ( header ):

    # header["Authorization"] = changetoken()
    header["User-Agent"] =changeagent()

    return header

def judge(r):
    if r==None:
        return False
    else:
        if r.status_code==403:
            time.sleep (2)
            return False
        if r.status_code==200:
            time.sleep(1)
            return True

@retry(stop_max_attempt_number=5,wait_random_max=12000)
def request_url ( url):
    agent=changeagent()
    header={"User-Agent":agent,
            "Connection":"close"}
    i= 0
    while i<5:
        try:
            s = requests.session()
            s.keep_alive = False
            # s.proxies = {"https": "112.64.233.130:9991"}
            r = s.get (url, headers=header,verify=False,timeout = 120)
            if judge(r):
                return  r

            elif r.status_code==404:
              return None

            else:
                header=change(header)
                i = i + 1

        except Exception as e:
                print(str(e))
                time.sleep(5)
    return None



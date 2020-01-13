import requests

a = input("댓글 작성: ")
url = 'https://comment-analysis.herokuapp.com/results'
r = requests.post(url,json={'ratio':5, 'comment':a})
isToxic = r.json()
isToxicText = ""
if isToxic == 1:
    isToxicText = "악플입니다."
else:
    isToxicText = "악플이 아닙니다."

print(isToxicText)
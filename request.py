import requests

a = input("댓글 작성: ")
url = 'https://comment-analysis.herokuapp.com/results'
r = requests.post(url,json={'ratio':5, 'comment':a})

print(r.json())
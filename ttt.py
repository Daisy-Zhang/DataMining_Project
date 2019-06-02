import jieba

def StringProcess(wds):
	mp = {}
	wd = jieba.cut(wds.strip())
	for w in wd:
		if not w in mp:
			mp[w] = 1
		else:
			mp[w] = mp[w] + 1
	return (wds, mp)


if __name__ == "__main__":
	s = '养老保险又新增两项农村老人人人可申领你领到了吗'
	s = StringProcess(s)
	print (s)
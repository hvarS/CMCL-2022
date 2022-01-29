def seperateHyphenToSentence(s):
    parts = s.split('-')
    return parts[-1]
def getLangText(s):
    parts = s.split('-')
    return parts[:-1]

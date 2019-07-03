from datetime import datetime


def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print('[' + display_now + ']' + ' ' + msg)
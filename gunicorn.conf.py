import os

port         = os.environ.get('PORT', '5000')
workers      = 2
worker_class = 'sync'
timeout      = 120
bind         = f'0.0.0.0:{port}'
accesslog    = '-'
errorlog     = '-'
loglevel     = 'info'
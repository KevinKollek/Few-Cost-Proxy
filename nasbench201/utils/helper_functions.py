import time

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{:}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def convert_secs2time(epoch_time, return_str=False):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    if return_str:
        return '{:02d}:{:02d}:{:02d}'.format(need_hour, need_mins, need_secs)
    else:
        return need_hour, need_mins, need_secs

def obtain_accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res
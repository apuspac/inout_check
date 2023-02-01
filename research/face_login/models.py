from django.db import models
from django.utils import timezone
import datetime

"""
class テーブル名(model):
    カラム名1 = データ型(その他の制約)
    カラム名2 = データ型(その他の制約)

"""

class person(models.Model):
    #id
    user_id  = models.CharField(max_length=128, verbose_name="ID")
    #名前
    user_name = models.CharField(max_length=128, verbose_name="名前")
    #ログイン時間
    login_time = models.ForeignKey('login_data', blank=True, null=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.user_id


# Create your models here.
class login_data(models.Model):
    in_out = (
        (1, 'in'),
        (0, 'out'),
    )

    user = models.ForeignKey('person', blank=True, null=True, on_delete=models.PROTECT)
    time= models.DateTimeField(default=datetime.datetime.now(), verbose_name="ログイン時間")
    status = models.IntegerField(choices=in_out, default=0, verbose_name="inout")

    def __str__(self):
        return str(self.user) + str(self.time) + str(self.status)

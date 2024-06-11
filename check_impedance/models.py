from django.db import models

# Create your models here.

class SaveDataModel(models.Model):
    tag = models.CharField(max_length=100)
    sampling_rate = models.IntegerField()
    main_data = models.JSONField()
    exe_data = models.JSONField()
    create = models.DateTimeField(auto_now=True,blank=True,null=True)
    first_name = models.CharField(max_length=200,blank=True,null=True)
    last_name = models.CharField(max_length=200,blank=True,null=True)
    age = models.CharField(max_length=3,blank=True,null=True)
    right_left_mix_hand = models.CharField(max_length=10,blank=True,null=True)
    gender = models.CharField(max_length=200,blank=True,null=True)
    phone = models.CharField(max_length=20,blank=True,null=True)
    national_code = models.CharField(max_length=15,blank=True,null=True)
    address = models.TextField(blank=True,null=True)
    
    def __str__(self):
        return self.tag
    
    
    

# class Profile(models.Model):

    

#     signals = models.ForeignKey(SaveDataModel,on_delete=models.CASCADE,blank=True,null=True)
    
#     def __str__(self):
#         return self.first_name + self.last_name
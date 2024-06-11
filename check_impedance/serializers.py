from rest_framework import serializers
from .models import SaveDataModel


class ConfigSerializers(serializers.Serializer):
    leadoff_mode = serializers.CharField()
    sampling_rate = serializers.IntegerField()
    
    
    
class SaveSignalSerializer(serializers.ModelSerializer):
  class Meta:
    model = SaveDataModel
    fields = ('tag',
              'sampling_rate',
              'main_data',
              'exe_data',
              "first_name",
              "last_name",
              "age",
              "right_left_mix_hand",
              "gender",
              "phone",
              "national_code",
              "address",)

    def create(self, validated_data):
        SaveDataModel.objects.create(**validated_data)





# vision_tracking

### Download Model File

Download SSD model trained on tianjin dataset:

链接: https://pan.baidu.com/s/1qZNLRmW 密码: fevw

Create a folder named models in the project folder, then download models files in models/SSD/tianjin/


Usage:

Change relative path of labelmap file to absolute path in .models/SSD/tianjin/deploy_300x300.prototxt:

save_output_param {
      label_map_file: "ABSOLUTE-PATH/models/SSD/tianjin/labelmap_tianjin.prototxt"
    }

Do tracking:

roslaunch vision_preception detection_and_tracking.launch

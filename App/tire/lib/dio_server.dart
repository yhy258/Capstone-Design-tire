
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

class Server with ChangeNotifier{
  late double accuracy;
  Future<Map> uploadImg(dynamic path) async {
    var dio = Dio();

    if (path != null) {
      var formData = FormData.fromMap({
        'image': await MultipartFile.fromFile(path),
      });

      final response = await dio.post(
          'http://seowtiredetect.ga/predict/image', data: formData); // 업로드 요청 // url 주소 받아와야 할듯
      var result = response.data;
      // var result = {"prediction" : "safety", "probability" : 0.6425};
      if(result == null){
        print("예외 발생~~");
        return result;
      }
      else{
        return result;
      }
      // else{
      //   print("예외 발생~~");
      //   return 0;
      // }
      // print(response.data.toString()); // 업로드 확인
    } else {
      //nothing
      print("path 없음 ");
      return {"prediction" : "error"};
      // return 0;
    }
  }
  Future<void> getResult() async {
    var dio = Dio();
    final response = await dio.get('http://seowtiredetect.ga/');
    print(response.data.toString());
  }
}

Server server = Server();
import "package:flutter/material.dart";
import 'package:image_picker/image_picker.dart';
import 'package:tire/dio_server.dart';
import 'dart:async';
import 'dart:io';
import 'package:tire/check_result.dart';
// import 'package:camera_camera/camera_camera.dart';

import 'package:tire/main.dart';

var black = const Color(0xff252525);

class ImgUpload extends StatefulWidget {
  const ImgUpload({Key? key}) : super(key: key);

  @override
  State<ImgUpload> createState() => _ImgUploadState();
}

class _ImgUploadState extends State<ImgUpload> {
  final ImagePicker _picker = ImagePicker();
  XFile? _image;
  String? _imagePath;
  dynamic source;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("이미지 첨부하기"),
        backgroundColor: black,
        // shadowColor: const Color(0xffF4F4F4),
      ),
      body: Center(
        child: Column(
          children: [
            const SizedBox(height: 200.0),
            IconButton(
              onPressed: () => takeImage(context),
              iconSize: 300,
              icon: Stack(children: <Widget>[
                Container(
                  // color: const Color(0xff252525),
                  decoration: _image != null
                      ? BoxDecoration(
                          border: Border.all(
                            style: BorderStyle.solid,
                            width: 3,
                            color: black,
                          ),
                          borderRadius: BorderRadius.circular(10),
                          image: DecorationImage(
                            image: FileImage(File(_imagePath!)),
                            fit: BoxFit.fill,
                          ),
                        )
                      : BoxDecoration(
                          border: Border.all(
                            style: BorderStyle.solid,
                            width: 5,
                            color: black,
                          ),
                          borderRadius: BorderRadius.circular(10),
                          image: DecorationImage(
                            colorFilter: ColorFilter.mode(
                                Colors.black.withOpacity(0.7),
                                BlendMode.srcATop),
                            image: const AssetImage(
                              "assets/guideTire.jpeg",
                            ),
                            fit: BoxFit.fill,
                          ),
                        ),
                ),
                _image != null ? Column() :
                 Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const <Widget>[
                    // SizedBox(height: 80.0),
                    Align(
                        alignment: Alignment.center,
                        child: Icon(
                          Icons.add,
                          size: 60,
                          color: Colors.white,
                        )),
                    Text("이와같은 타이어 사진을 업로드 해주세요 ",
                        style: TextStyle(
                          fontSize: 15,
                          height: 1.5,
                          color: Colors.white,
                        )),
                  ],
                )
              ]),
            ),
            // const SizedBox(height: 10.0),
            Padding(
              padding: const EdgeInsets.all(15),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(primary: black),
                onPressed: () async {
                  var check = await server.uploadImg(_imagePath);
                  // print(check);
                  // var m_probability = check["probability"];
                  // m_probability = toString(m_probability);
                  if (check["prediction"] == 'safety') {
                    MaterialPageRoute route = MaterialPageRoute(
                        builder: (context) => prd_safety(
                            probability:
                                (check["probability"] * 100).toString()));
                    Navigator.push(context, route);
                  } else if (check["prediction"] == 'danger') {
                    MaterialPageRoute route = MaterialPageRoute(
                        builder: (context) => prd_danger(
                            probability:
                                (check["probability"] * 100).toString()));
                    Navigator.push(context, route);
                  } else {
                    // 사진 에러
                    print(check);
                    print('사진에러');
                  }
                  // server.uploadImg(_imagePath);
                },
                child: const Text("검사하기"),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future takeImage(mContext) {
    return showDialog(
        context: mContext,
        builder: (context) {
          return SimpleDialog(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            title: const Text('이미지 첨부',
                style: TextStyle(
                  color: Colors.black,
                  fontWeight: FontWeight.bold,
                )),
            children: <Widget>[
              SimpleDialogOption(
                  child: const Text(
                    '카메라로 찍기',
                    style: TextStyle(color: Colors.black),
                  ),
                  onPressed: () => _pickImg(ImageSource.camera, context)),
              SimpleDialogOption(
                  child: const Text(
                    '갤러리에서 가져오기',
                    style: TextStyle(color: Colors.black),
                  ),
                  onPressed: () => _pickImg(ImageSource.gallery, context)),
              SimpleDialogOption(
                  child: const Text(
                    '취소',
                    style: TextStyle(color: Colors.black),
                  ),
                  onPressed: () => Navigator.pop(context)),
            ],
          );
        });
  }

  Future<dynamic> _pickImg(ImageSource source, context) async {
    // source = sources;
    Navigator.pop(context);
    XFile? image = await _picker.pickImage(
      source: source,
      maxHeight: 240,
      maxWidth: 240,
    );

    if (mounted) {
      // print("여기대ㅏㅣㅏㅏ");
      if (image != null) {
        setState(() {
          _image = image;
          _imagePath = _image?.path;
        });
      }
    }
  }
}

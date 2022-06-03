// import 'package:camera_camera/camera_camera.dart';
import 'package:tire/img_upload.dart';
import 'package:simple_shadow/simple_shadow.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';



void main() => runApp(Myapp());
var black = const Color(0xff252525);

class Myapp extends StatelessWidget {
  // const Myapp({Key? key}) : super(key: key);
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      title: "Image camera",
      // theme: new ThemeData(scaffoldBackgroundColor: const Color(0xFFF4F4F4)),
      home: MyHomePage(title:"title")
    );
  }
}
class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, this.title}) : super(key: key);

  final String? title;
  @override
  State<MyHomePage> createState() => _MyHomePageState();
  // _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  // final photos = <file>[];

  // final ImagePicker _picker = ImagePicker();
  // XFile? _image;
  // dynamic? _imagePath;

  @override
  Widget build(BuildContext context) {
    Size screenSize = MediaQuery.of(context).size;
    return  Scaffold(
      backgroundColor: const Color(0xffF4F4F4),
       body: SafeArea(
         child: Column(
           children: <Widget>[
             const SizedBox(height:150.0),
             SimpleShadow(
                 child: Image.asset("assets/mainCar.png",width: 150,height:150),
               opacity: 0.5,
               offset: Offset(2,2),
               sigma: 5,
             ),
             // CameraCamera(
             //     onFile: (file) => print(file),
             // ),
             Container(
                 decoration: const BoxDecoration(
                     color: Colors.blue,
                     borderRadius: BorderRadius.all(
                         Radius.circular(10)
                     )
                 )
             ),

             const SizedBox(height:50.0),
             const Text("CHECK YOUR TIRE WEAR",
               textAlign: TextAlign.center,
               style: TextStyle(fontWeight: FontWeight.bold,
               fontSize: 16,
                 shadows: <Shadow>[
                   Shadow(
                     offset: Offset(1.0, 1.0),
                     blurRadius:3.0,
                   )
                 ],
               )
             ),
             const Text("\n- MADE BY TEAM JUNHUK -",
               textAlign: TextAlign.center,
               style: TextStyle(fontStyle: FontStyle.italic),
             ),
             SizedBox(
               height: screenSize.height * 0.2,
               width: screenSize.width,
               // color: Color (0xff252525 ),
               child: CustomPaint(
                 painter: Curveline(),
               ),

             ),
             ElevatedButton(
               style: ElevatedButton.styleFrom(
                  primary: black
               ),
               onPressed:() {
                 MaterialPageRoute route = MaterialPageRoute(builder: (context) => const ImgUpload());
                Navigator.push(
                context, route);
               },
               child: const Text("검사하기"),

             ),
           ],
         ),
       )
     );
  }
}

class Curveline extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final startSize = Size(0, size.height * 0.5);
    // TODO: implement paint
    final endSize = Size(size.width , size.height * 0.5);

    final linePath = Path()
      ..moveTo(0,startSize.height)
      ..quadraticBezierTo(startSize.width + endSize.width / 2, size.height, endSize.width, endSize.height);

    final paint = Paint()
        ..color = black
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke;

    canvas.drawPath(linePath,paint);
  }

  @override
  // bool shouldRepaint(covariant CustomPainter oldDelegate) {
  //   // TODO: implement shouldRepaint
  //   throw UnimplementedError();
  // }
  bool shouldRepaint(CustomPainter oldDelegate) => false;

}





import "package:flutter/material.dart";
import 'package:tire/main.dart';


var black = const Color(0xff252525);

class prd_safety extends StatefulWidget {
  final String probability;
  prd_safety({Key? key, required this.probability}) : super(key: key);

  @override
  State<prd_safety> createState() => _prd_safetyState(probability: probability);
}

class _prd_safetyState extends State<prd_safety> {
  final String probability;
  _prd_safetyState({Key? key, required this.probability});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: black,
      ),
      body: Container(
        child: Column(
            children:<Widget> [
              const SizedBox(height: 200.0),
              Center(
                  child: Image.asset("assets/Smiley.png",height:150)),
              const Center(
                  child: Text(" 안전한 타이어 입니다!",
                      // textAlign: TextAlign.center,
                      style: TextStyle(fontWeight: FontWeight.bold,
                          fontSize:20,
                          height: 1.5,
                      ),
                      )
              ),
              // Center(
              //     child: Text("Confidence: "+ probability,
              //       style: const TextStyle(
              //         fontSize:20,
              //         height: 1.5,
              //       ),
              //     )
              // ),
              const SizedBox(height: 30.0),
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                    primary: black
                ),
                onPressed:() {
                  //   MaterialPageRoute route = MaterialPageRoute(builder: (context) => Myapp());
                  //   Navigator.push(
                  //       context, route);
                  // },
                  Navigator.pushNamedAndRemoveUntil(context, '/', (_) => false);
                  Navigator.push(
                    context, MaterialPageRoute(builder: (context) => Myapp()),);
                  // setState(() {
                    // img_upload.
                  // });
                },
                  child: const Text("검사하기"),
              ),
            ]
        ),
      ),
    );
  }
}

class prd_danger extends StatefulWidget {
  final String probability;
  prd_danger({Key? key, required this.probability}) : super(key: key);

  @override
  State<prd_danger> createState() => _prd_dangerState(probability: probability);
}

class _prd_dangerState extends State<prd_danger> {
  final String probability;
  _prd_dangerState({Key? key, required this.probability});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(backgroundColor: black),
      body: Container(
        child: Column(
          children:<Widget> [
            const SizedBox(height: 200.0),
            Center(
                child: Image.asset("assets/Smiley Sad.png",height:150)),
            const Center(
                child: Text("타이어를 교체해주세요!",
                  style: TextStyle(fontWeight: FontWeight.bold,
                    fontSize:20,
                    height: 1.5,
                  ),
                )
            ),
            // Center(
            //     child: Text("Confidence: "+ probability,
            //       style: const TextStyle(
            //         fontSize:20,
            //         height: 1.5,
            //       ),
            //     )
            // ),
            const SizedBox(height: 30.0),
        ElevatedButton(
        style: ElevatedButton.styleFrom(
            primary: black
        ),
        onPressed:() {
          MaterialPageRoute route = MaterialPageRoute(builder: (context) => Myapp());
          Navigator.push(
              context, route);
        },
        child: const Text("검사하기"),
      ),
      ]
        ),
      ),
    );
  }
}


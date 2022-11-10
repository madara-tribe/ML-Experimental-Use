import SwiftUI
import CoreML
import Vision
import PhotosUI

struct ContentView: View {
    // config
    @State var classficationLabel = ""
    @State var predtime = ""
    @State var captureImage: UIImage? = nil
    //control cap screen situatuion
    @State var isShowSheet = false
    var ratio:Double = 0.8

    // Request
    func CreateRequest()->VNCoreMLRequest{
        do {
            // Model instance
            let start_time = Date()
            let configuration = MLModelConfiguration()
            let model = try VNCoreMLModel(for: Resnet50(configuration:configuration).model)
            // create Request
            let request = VNCoreMLRequest(model:model, completionHandler:{request, error in
                // post proccesing
                ModelPrediction(request: request)
                
            })
            // Prediction Latency
            predtime = calcurateTime(stime: start_time)
            return request
        } catch {
            fatalError("cat not load model")
        }
    }
    
    
    // Model Prediction
    func ModelPrediction(request: VNRequest){
        // get results from prediction
        guard let results = request.results else{
            return
        }
        let classification = results as! [VNClassificationObservation]
        // get results label
        classficationLabel = classification[0].identifier
    }
    
    // Handler
    func Handler(image:UIImage){
        // conver UIImage to CIImage
        guard let ciImage = CIImage(image : image) else {
            fatalError("can not convert to CIImage")
        }
        // handler instance
        let handler = VNImageRequestHandler(ciImage: ciImage)
        // request
        let classificationRequest = CreateRequest()
        // do handler
        do {
            try handler.perform([classificationRequest])
        } catch {
            fatalError("failed to predict image")
        }
    }
    
    func run_trimmingImage(_ image: UIImage){
        captureImage = trimmingImage(image, ratio_:ratio)
    }
    
    // main
    var body: some View {
        VStack {
            Spacer()
            if let unwrapCaptureImage = captureImage{
                // plot ui or cap ci image
                Image(uiImage: unwrapCaptureImage)
                    .resizable()
                .aspectRatio(contentMode: .fit)} else {
                Image("Noimg")
            }
            Text(classficationLabel)
                .padding()
                .font(.title)
            Text(predtime)
                .padding()
                .font(.title)
            Button("trimming"){
                run_trimmingImage(captureImage!)
            }
            .frame(height: 50)
            .multilineTextAlignment(.center)
            Button("Start Pedict"){
                Handler(image:captureImage!)
            }
            Button(action:{
                // action after push
                if UIImagePickerController.isSourceTypeAvailable(.photoLibrary){
                    print("select image")
                    isShowSheet = true
                }
            }){
                Text("Select imagee")
                    .frame(maxWidth: .infinity)
                    .frame(height: 50)
                    .multilineTextAlignment(.center)
                    .background(Color.blue)
                    .foregroundColor(Color.white)
            }
            .sheet(isPresented:$isShowSheet){
                PHPickerView(
                    isShowSheet: $isShowSheet,
                    captureImage: $captureImage)
            }
        } // VStack last block
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

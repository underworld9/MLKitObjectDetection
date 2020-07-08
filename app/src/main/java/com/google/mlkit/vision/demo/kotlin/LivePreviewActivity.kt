/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.kotlin

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.android.gms.common.annotation.KeepName
import com.google.mlkit.common.model.LocalModel
import com.google.mlkit.vision.demo.CameraSource
import com.google.mlkit.vision.demo.CameraSourcePreview
import com.google.mlkit.vision.demo.GraphicOverlay
import com.google.mlkit.vision.demo.R
import com.google.mlkit.vision.demo.kotlin.objectdetector.ObjectDetectorProcessor
import com.google.mlkit.vision.objects.ObjectDetectorOptionsBase
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions

/** Live preview demo for ML Kit APIs.  */
@KeepName
class LivePreviewActivity : AppCompatActivity(), ActivityCompat.OnRequestPermissionsResultCallback {

    private var cameraSource: CameraSource? = null
    private var preview: CameraSourcePreview? = null
    private var graphicOverlay: GraphicOverlay? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_live_preview)

        preview = findViewById(R.id.preview)

        graphicOverlay = findViewById(R.id.graphic_overlay)

        createCameraSource()
    }

    private fun createCameraSource() {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = CameraSource(this, graphicOverlay)
        }
        val localModel = LocalModel.Builder().setAssetFilePath("custom_models/modemHeadsetAUTOML.tflite").build()

        val builder = CustomObjectDetectorOptions.Builder(localModel).setDetectorMode(ObjectDetectorOptionsBase.STREAM_MODE)
                .enableClassification()
        val customObjectDetectorOptions = builder.build()

        cameraSource!!.setMachineLearningFrameProcessor(ObjectDetectorProcessor(this, customObjectDetectorOptions)
        )
    }

    public override fun onResume() {
        super.onResume()
        createCameraSource()
        if (cameraSource != null) {
            preview!!.start(cameraSource, graphicOverlay)
        }
    }

    /** Stops the camera.  */
    override fun onPause() {
        super.onPause()
        preview?.stop()
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (cameraSource != null) {
            cameraSource?.release()
        }
    }
}

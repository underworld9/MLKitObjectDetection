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

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.annotation.GuardedBy
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.TaskExecutors
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.demo.*
import com.google.mlkit.vision.demo.preference.PreferenceUtils
import java.nio.ByteBuffer
import java.util.*

/**
 * Abstract base class for ML Kit frame processors. Subclasses need to implement {@link
 * #onSuccess(T, FrameMetadata, GraphicOverlay)} to define what they want to with the detection
 * results and {@link #detectInImage(VisionImage)} to specify the detector object.
 *
 * @param <T> The type of the detected feature.
 */
abstract class VisionProcessorBase<T>(context: Context) : VisionImageProcessor {

    companion object {
        private const val TAG = "VisionProcessorBase"
    }

    private var activityManager: ActivityManager =
            context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    private val fpsTimer = Timer()
    private val executor = ScopedExecutor(TaskExecutors.MAIN_THREAD)

    // Whether this processor is already shut down
    private var isShutdown = false

    // Used to calculate latency, running in the same thread, no sync needed.
    private var numRuns = 0
    private var totalRunMs: Long = 0
    private var maxRunMs: Long = 0
    private var minRunMs = Long.MAX_VALUE

    // Frame count that have been processed so far in an one second interval to calculate FPS.
    private var frameProcessedInOneSecondInterval = 0
    private var framesPerSecond = 0

    // To keep the latest images and its metadata.
    @GuardedBy("this")
    private var latestImage: ByteBuffer? = null

    @GuardedBy("this")
    private var latestImageMetaData: FrameMetadata? = null

    // To keep the images and metadata in process.
    @GuardedBy("this")
    private var processingImage: ByteBuffer? = null

    @GuardedBy("this")
    private var processingMetaData: FrameMetadata? = null

    init {
        fpsTimer.scheduleAtFixedRate(
                object : TimerTask() {
                    override fun run() {
                        framesPerSecond = frameProcessedInOneSecondInterval
                        frameProcessedInOneSecondInterval = 0
                    }
                },
                0,
                1000
        )
    }

    // -----------------Code for processing live preview frame from Camera1 API-----------------------
    @Synchronized
    override fun processByteBuffer(
            data: ByteBuffer?,
            frameMetadata: FrameMetadata?,
            graphicOverlay: GraphicOverlay
    ) {
        latestImage = data
        latestImageMetaData = frameMetadata
        if (processingImage == null && processingMetaData == null) {
            processLatestImage(graphicOverlay)
        }
    }

    @Synchronized
    private fun processLatestImage(graphicOverlay: GraphicOverlay) {
        processingImage = latestImage
        processingMetaData = latestImageMetaData
        latestImage = null
        latestImageMetaData = null
        if (processingImage != null && processingMetaData != null && !isShutdown) {
            processImage(processingImage!!, processingMetaData!!, graphicOverlay)
        }
    }

    private fun processImage(
            data: ByteBuffer,
            frameMetadata: FrameMetadata,
            graphicOverlay: GraphicOverlay
    ) {
        // If live viewport is on (that is the underneath surface view takes care of the camera preview
        // drawing), skip the unnecessary bitmap creation that used for the manual preview drawing.
        val bitmap =
                if (PreferenceUtils.isCameraLiveViewportEnabled(graphicOverlay.context)) null
                else BitmapUtils.getBitmap(data, frameMetadata)
        requestDetectInImage(
                InputImage.fromByteBuffer(
                        data,
                        frameMetadata.width,
                        frameMetadata.height,
                        frameMetadata.rotation,
                        InputImage.IMAGE_FORMAT_NV21
                ),
                graphicOverlay,
                bitmap
        )
                .addOnSuccessListener(executor) { processLatestImage(graphicOverlay) }
    }

    // -----------------Common processing logic-------------------------------------------------------
    private fun requestDetectInImage(
            image: InputImage,
            graphicOverlay: GraphicOverlay,
            originalCameraImage: Bitmap?
    ): Task<T> {
        val startMs = SystemClock.elapsedRealtime()
        return detectInImage(image).addOnSuccessListener(executor) { results: T ->
            val currentLatencyMs = SystemClock.elapsedRealtime() - startMs
            numRuns++
            frameProcessedInOneSecondInterval++
            totalRunMs += currentLatencyMs
            maxRunMs = currentLatencyMs.coerceAtLeast(maxRunMs)
            minRunMs = currentLatencyMs.coerceAtMost(minRunMs)
            // Only log inference info once per second. When frameProcessedInOneSecondInterval is
            // equal to 1, it means this is the first frame processed during the current second.
            if (frameProcessedInOneSecondInterval == 1) {
                Log.d(TAG, "Max latency is: $maxRunMs")
                Log.d(TAG, "Min latency is: $minRunMs")
                Log.d(
                        TAG,
                        "Num of Runs: " + numRuns + ", Avg latency is: " + totalRunMs / numRuns
                )
                val mi = ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(mi)
                val availableMegs = mi.availMem / 0x100000L
                Log.d(
                        TAG,
                        "Memory available in system: $availableMegs MB"
                )
            }
            graphicOverlay.clear()
            if (originalCameraImage != null) {
                graphicOverlay.add(
                        CameraImageGraphic(
                                graphicOverlay,
                                originalCameraImage
                        )
                )
            }
            this@VisionProcessorBase.onSuccess(results, graphicOverlay)
            graphicOverlay.add(
                    InferenceInfoGraphic(
                            graphicOverlay,
                            currentLatencyMs.toDouble(),
                            framesPerSecond
                    )
            )
            graphicOverlay.postInvalidate()
        }
                .addOnFailureListener(executor) { e: Exception ->
                    graphicOverlay.clear()
                    graphicOverlay.postInvalidate()
                    Toast.makeText(
                            graphicOverlay.context,
                            "Failed to process.\nError: " +
                                    e.localizedMessage +
                                    "\nCause: " +
                                    e.cause,
                            Toast.LENGTH_LONG
                    )
                            .show()
                    e.printStackTrace()
                    this@VisionProcessorBase.onFailure(e)
                }
    }

    override fun stop() {
        executor.shutdown()
        isShutdown = true
        numRuns = 0
        totalRunMs = 0
        fpsTimer.cancel()
    }

    protected abstract fun detectInImage(image: InputImage): Task<T>

    protected abstract fun onSuccess(results: T, graphicOverlay: GraphicOverlay)

    protected abstract fun onFailure(e: Exception)
}

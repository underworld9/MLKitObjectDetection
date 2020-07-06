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

package com.google.mlkit.vision.demo.preference;

import android.content.Context;
import android.content.SharedPreferences;
import android.preference.PreferenceManager;

import androidx.annotation.Nullable;
import androidx.annotation.StringRes;

import com.google.android.gms.common.images.Size;
import com.google.common.base.Preconditions;
import com.google.mlkit.common.model.LocalModel;
import com.google.mlkit.vision.demo.CameraSource;
import com.google.mlkit.vision.demo.CameraSource.SizePair;
import com.google.mlkit.vision.demo.R;
import com.google.mlkit.vision.objects.custom.CustomObjectDetectorOptions;

import java.util.Objects;

/**
 * Utility class to retrieve shared preferences.
 */
public class PreferenceUtils {

    static void saveString(Context context, @StringRes int prefKeyId, @Nullable String value) {
        PreferenceManager.getDefaultSharedPreferences(context)
                .edit()
                .putString(context.getString(prefKeyId), value)
                .apply();
    }

    @Nullable
    public static SizePair getCameraPreviewSizePair(Context context, int cameraId) {
        Preconditions.checkArgument(
                cameraId == CameraSource.CAMERA_FACING_BACK
                        || cameraId == CameraSource.CAMERA_FACING_FRONT);
        String previewSizePrefKey;
        String pictureSizePrefKey;
        if (cameraId == CameraSource.CAMERA_FACING_BACK) {
            previewSizePrefKey = context.getString(R.string.pref_key_rear_camera_preview_size);
            pictureSizePrefKey = context.getString(R.string.pref_key_rear_camera_picture_size);
        } else {
            previewSizePrefKey = context.getString(R.string.pref_key_front_camera_preview_size);
            pictureSizePrefKey = context.getString(R.string.pref_key_front_camera_picture_size);
        }

        try {
            SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
            return new SizePair(
                    Size.parseSize(Objects.requireNonNull(sharedPreferences.getString(previewSizePrefKey, null))),
                    Size.parseSize(Objects.requireNonNull(sharedPreferences.getString(pictureSizePrefKey, null))));
        } catch (Exception e) {
            return null;
        }
    }

    public static CustomObjectDetectorOptions getCustomObjectDetectorOptionsForLivePreview(
            Context context, LocalModel localModel) {
        return getCustomObjectDetectorOptions(
                context,
                localModel
        );
    }

    private static CustomObjectDetectorOptions getCustomObjectDetectorOptions(
            Context context,
            LocalModel localModel) {

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);

        boolean enableMultipleObjects =
                sharedPreferences.getBoolean(context.getString(R.string.pref_key_live_preview_object_detector_enable_multiple_objects), false);
        boolean enableClassification =
                sharedPreferences.getBoolean(context.getString(R.string.pref_key_live_preview_object_detector_enable_classification), true);

        CustomObjectDetectorOptions.Builder builder =
                new CustomObjectDetectorOptions.Builder(localModel).setDetectorMode(com.google.mlkit.vision.objects.ObjectDetectorOptionsBase.STREAM_MODE);
        if (enableMultipleObjects) {
            builder.enableMultipleObjects();
        }
        if (enableClassification) {
            builder.enableClassification();
        }
        return builder.build();
    }

    public static boolean isCameraLiveViewportEnabled(Context context) {
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(context);
        String prefKey = context.getString(R.string.pref_key_camera_live_viewport);
        return sharedPreferences.getBoolean(prefKey, false);
    }
}

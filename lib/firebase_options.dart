// File generated by FlutterFire CLI.
// ignore_for_file: lines_longer_than_80_chars, avoid_classes_with_only_static_members
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show defaultTargetPlatform, kIsWeb, TargetPlatform;

/// Default [FirebaseOptions] for use with your Firebase apps.
///
/// Example:
/// ```dart
/// import 'firebase_options.dart';
/// // ...
/// await Firebase.initializeApp(
///   options: DefaultFirebaseOptions.currentPlatform,
/// );
/// ```
class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) {
      return web;
    }
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      case TargetPlatform.macOS:
        return macos;
      case TargetPlatform.windows:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for windows - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      case TargetPlatform.linux:
        throw UnsupportedError(
          'DefaultFirebaseOptions have not been configured for linux - '
          'you can reconfigure this by running the FlutterFire CLI again.',
        );
      default:
        throw UnsupportedError(
          'DefaultFirebaseOptions are not supported for this platform.',
        );
    }
  }

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: 'AIzaSyCfYX6fZvwFVHLAqKI-0rGv6y0IUcP01CU',
    appId: '1:308505602070:web:d65323fc46c09938d92d37',
    messagingSenderId: '308505602070',
    projectId: 'safedrive-12f49',
    authDomain: 'safedrive-12f49.firebaseapp.com',
    storageBucket: 'safedrive-12f49.appspot.com',
    measurementId: 'G-Z1FQ02QC97',
  );

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyCRy3gU7QcwvJsoxZTkBjqAxuaxlyqK8ds',
    appId: '1:308505602070:android:705b6bc270806be2d92d37',
    messagingSenderId: '308505602070',
    projectId: 'safedrive-12f49',
    storageBucket: 'safedrive-12f49.appspot.com',
  );

  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: 'AIzaSyDLgxBuLInMU29at8DeNkpqwa-SzaqkJZI',
    appId: '1:308505602070:ios:def41db9c5e6938ed92d37',
    messagingSenderId: '308505602070',
    projectId: 'safedrive-12f49',
    storageBucket: 'safedrive-12f49.appspot.com',
    iosClientId: '308505602070-s8eonnhrpob6prh7jogo7e5kh9tti6hr.apps.googleusercontent.com',
    iosBundleId: 'com.example.admin',
  );

  static const FirebaseOptions macos = FirebaseOptions(
    apiKey: 'AIzaSyDLgxBuLInMU29at8DeNkpqwa-SzaqkJZI',
    appId: '1:308505602070:ios:def41db9c5e6938ed92d37',
    messagingSenderId: '308505602070',
    projectId: 'safedrive-12f49',
    storageBucket: 'safedrive-12f49.appspot.com',
    iosClientId: '308505602070-s8eonnhrpob6prh7jogo7e5kh9tti6hr.apps.googleusercontent.com',
    iosBundleId: 'com.example.admin',
  );
}

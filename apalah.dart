import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Child Tablet',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        fontFamily: 'Roboto', // Font yang lebih clean
      ),
      home: ChildTabletScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ChildTabletScreen extends StatefulWidget {
  @override
  _ChildTabletScreenState createState() => _ChildTabletScreenState();
}

class _ChildTabletScreenState extends State<ChildTabletScreen> {
  bool isActive = true;
  int batteryLevel = 90;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          image: DecorationImage(
            image: AssetImage('assets/wl.png'),
            fit: BoxFit.cover,
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: EdgeInsets.all(16.0),
            child: Column(
              children: [
                // Header dengan status tablet
                _buildHeader(),

                SizedBox(height: 20),

                // Menu grid - diperbesar untuk memenuhi layar
                Expanded(
                  flex: 3, // Memberikan lebih banyak ruang untuk menu
                  child: _buildMenuGrid(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      margin: EdgeInsets.only(bottom: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Status Tablet Kiri - diperbesar
          Container(
            width: MediaQuery.of(context).size.width * 0.4, // hampir setengah layar
            padding: EdgeInsets.symmetric(horizontal: 24, vertical: 18),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(25),
              border: Border.all(color: Colors.blue.shade300, width: 2),
            ),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: Colors.green.shade400,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.tablet_android,
                    color: Colors.white,
                    size: 24,
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        'Child Tablet',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          color: Colors.black87,
                        ),
                      ),
                      Row(
                        children: [
                          Text(
                            'Active',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey.shade600,
                            ),
                          ),
                          SizedBox(width: 8),
                          Icon(
                            Icons.battery_std,
                            color: Colors.green.shade400,
                            size: 16,
                          ),
                          Text(
                            '$batteryLevel %',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey.shade600,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          // Pesan Selamat - diperbesar
          Container(
            width: MediaQuery.of(context).size.width * 0.45, // hampir setengah layar
            padding: EdgeInsets.symmetric(horizontal: 24, vertical: 18),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(25),
              border: Border.all(color: Colors.blue.shade300, width: 2),
            ),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: Colors.green.shade400,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(
                    Icons.child_care,
                    color: Colors.white,
                    size: 24,
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: Text(
                    'Saya mau boxing, ibu',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                      color: Colors.black87,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMenuGrid() {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Sebelah kiri kosong
        Expanded(
          flex: 1,
          child: Container(),
        ),

        // Sebelah kanan untuk menu cards
        Expanded(
          flex: 1,
          child: Column(
            children: [
              // Row pertama - tinggi diperbesar
              Expanded(
                flex: 3, // Diperbesar dari 2 menjadi 3 untuk memberikan lebih banyak ruang vertikal
                child: Row(
                  children: [
                    Expanded(
                      child: _buildMenuCard(
                        title: 'Add\nWord',
                        subtitle: 'Word\nImage\nSound',
                        color: Colors.blue.shade100,
                        onTap: () => _showFeatureDialog('Add Word'),
                      ),
                    ),
                    SizedBox(width: 16), // Jarak antar kotak diperbesar
                    Expanded(
                      child: _buildMenuCard(
                        title: 'Pin\nWord',
                        subtitle: '1000 Word',
                        color: Colors.pink.shade100,
                        onTap: () => _showFeatureDialog('Pin Word'),
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 20), // Jarak antar row diperbesar dari 16 menjadi 20
              // Row kedua - tinggi diperbesar
              Expanded(
                flex: 3, // Diperbesar dari 2 menjadi 3 untuk memberikan lebih banyak ruang vertikal
                child: Row(
                  children: [
                    Expanded(
                      child: _buildMenuCard(
                        title: 'Edit\nWord',
                        subtitle: 'Word\nImage\nSound',
                        color: Colors.green.shade100,
                        onTap: () => _showFeatureDialog('Edit Word'),
                      ),
                    ),
                    SizedBox(width: 16), // Jarak antar kotak diperbesar
                    Expanded(
                      child: _buildMenuCard(
                        title: 'Erase\nWord',
                        subtitle: 'Word\nImage\nSound',
                        color: Colors.yellow.shade100,
                        onTap: () => _showFeatureDialog('Erase Word'),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMenuCard({
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        // Menghapus height yang fixed agar mengikuti ruang yang tersedia
        decoration: BoxDecoration(
          color: color,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 6,
              offset: Offset(0, 2),
            ),
          ],
        ),
        child: Padding(
          padding: EdgeInsets.all(16), // Diperbesar dari 12 menjadi 16
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                title,
                style: TextStyle(
                  fontSize: 18, // Diperbesar dari 16 menjadi 18
                  fontWeight: FontWeight.w700,
                  color: Colors.black87,
                  height: 1.1,
                ),
              ),
              Text(
                subtitle,
                style: TextStyle(
                  fontSize: 12, // Diperbesar dari 10 menjadi 12
                  color: Colors.black54,
                  height: 1.2,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  void _showFeatureDialog(String featureName) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          title: Row(
            children: [
              Container(
                width: 32,
                height: 32,
                decoration: BoxDecoration(
                  color: Colors.green.shade400,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  Icons.child_care,
                  color: Colors.white,
                  size: 18,
                ),
              ),
              SizedBox(width: 12),
              Text(
                featureName,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          content: Text(
            'Fitur $featureName akan segera tersedia!\n\nAnak dapat belajar kata-kata baru dengan cara yang menyenangkan.',
            style: TextStyle(fontSize: 14),
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Container(
                padding: EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.blue.shade100,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  'OK',
                  style: TextStyle(
                    color: Colors.black87,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}
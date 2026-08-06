using System;
using System.Linq;
using System.Text.RegularExpressions;
using Inlite.ClearImageNet;

class Program
{
    // 覆盖你出现过的两类 SN：
    // 4E25A0175936 -> 6位字母数字 + 6位数字
    // 21500872904ERA001005 / 21500108432SRA501316 -> 11位数字 + 3字母 + 6位数字
    static readonly Regex[] SnRegs =
    {
        new Regex(@"^[0-9A-Z]{6}\d{6}$", RegexOptions.Compiled),
        new Regex(@"^\d{11}[A-Z]{3}\d{6}$", RegexOptions.Compiled),
    };

    static string Clean(string s) =>
        Regex.Replace(s ?? "", @"[^0-9A-Za-z]", "").ToUpperInvariant();

    static bool IsSn(string s) => SnRegs.Any(r => r.IsMatch(s));

    static void ConfigureReader(BarcodeReader reader, bool auto1d, bool debugShowEan)
    {
        // 你关心的 1D（属性名来自官方属性表）:contentReference[oaicite:2]{index=2}
        reader.Code128 = true;
        reader.Ucc128  = true;
        reader.Code39  = true;

        // 平时：关掉 EAN/UPC，避免 13 位纯数字污染结果
        // debug 时：可以临时打开，让你确认“引擎到底有没有读到条码”
        reader.Ean13 = debugShowEan;
        reader.Ean8  = debugShowEan;
        reader.Upca  = debugShowEan;
        reader.Upce  = debugShowEan;

        // 方向别关死（轻微倾斜也能读）；这些属性同样在官方属性表里 :contentReference[oaicite:3]{index=3}
        reader.Horizontal = true;
        reader.Vertical   = true;
        reader.Diagonal   = true;

        // Auto1D=true：自动找条码类型，慢但更容易命中（官方也这么说）:contentReference[oaicite:4]{index=4}
        reader.Auto1D = auto1d;

        // 多条码场景可以先不限制
        reader.MaxBarcodes = 0;
    }

    static void RepairPage(ImageEditor editor)
    {
        // 官方“优化识别”示例里常用的一串修复（你可以把它当作第三道兜底）:contentReference[oaicite:5]{index=5}
        editor.AutoDeskew();   // 基于内容纠偏 :contentReference[oaicite:6]{index=6}
        editor.AutoRotate();
        editor.AutoCrop(10, 10, 10, 10);
        // 背景复杂时二值化有帮助；如果你发现变差，可以把这行注释掉
        editor.ToBitonal();

        // BorderExtract 不一定每张都有边框，可能抛异常，所以包一下
        try { editor.BorderExtract(BorderExtractMode.DeskewCrop); } catch { }

        editor.SmoothCharacters();
        editor.CleanNoise(3);
        editor.ReconstructLines(LineDirection.HorzAndVert);
    }

    static string PickSnOrNull(Barcode[] barcodes, bool debug)
    {
        if (debug)
        {
            foreach (var b in barcodes ?? Array.Empty<Barcode>())
                Console.Error.WriteLine($"[DBG] type={b.Type} text={b.Text}");
        }

        foreach (var b in barcodes ?? Array.Empty<Barcode>())
        {
            var txt = Clean(b.Text);
            if (IsSn(txt)) return txt;
        }
        return null;
    }

    static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.WriteLine("");
            return 2;
        }

        bool debug = args.Contains("--debug");
        string file = args.First(a => a != "--debug");
        int page = 1;

        try
        {
            // Pass 1：直接读文件（最不破坏，官方示例就是 reader.Read(filename, page)）:contentReference[oaicite:7]{index=7}
            using (var reader = new BarcodeReader())
            {
                ConfigureReader(reader, auto1d: false, debugShowEan: debug);
                var barcodes = reader.Read(file, page);
                var sn = PickSnOrNull(barcodes, debug);
                if (sn != null) { Console.WriteLine(sn); return 0; }
                if (debug) Console.Error.WriteLine("[DBG] pass1: direct read auto1d=false -> no SN");
            }

            // Pass 2：Auto1D=true 兜底（慢但更容易命中）:contentReference[oaicite:8]{index=8}
            using (var reader = new BarcodeReader())
            {
                ConfigureReader(reader, auto1d: true, debugShowEan: debug);
                var barcodes = reader.Read(file, page);
                var sn = PickSnOrNull(barcodes, debug);
                if (sn != null) { Console.WriteLine(sn); return 0; }
                if (debug) Console.Error.WriteLine("[DBG] pass2: direct read auto1d=true -> no SN");
            }

            // Pass 3：修复后再读（官方优化示例思路）:contentReference[oaicite:9]{index=9}
            using (var editor = new ImageEditor())
            {
                editor.Image.Open(file, page);
                RepairPage(editor);

                using (var reader = new BarcodeReader())
                {
                    ConfigureReader(reader, auto1d: true, debugShowEan: debug);
                    var barcodes = reader.Read(editor);
                    var sn = PickSnOrNull(barcodes, debug);
                    if (sn != null) { Console.WriteLine(sn); return 0; }
                    if (debug) Console.Error.WriteLine("[DBG] pass3: editor+repair -> no SN");
                }
            }

            // 没找到 SN：输出空行（stdout 保持干净给 Python 用）
            Console.WriteLine("");
            return 1;
        }
        catch (Exception ex)
        {
            Console.WriteLine("ERROR: " + ex.Message);
            return 3;
        }
    }
}

package edu.pengli.nlp.platform.util;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;

public class FileOperation {

	public static void copyFile(File originalDir, File desinationDir,
			String fileName) {
		try {

			BufferedReader in = getBufferedReader(originalDir, fileName);
			PrintWriter out = getPrintWriter(desinationDir, fileName);

			String input = null;
			while ((input = in.readLine()) != null) {
				out.println(input);
			}
			in.close();
			out.close();
		} catch (FileNotFoundException ex) {
			System.out
					.println(ex.getMessage() + " in the specified directory.");
			System.exit(0);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}
	
	public static void copyFile(String originalPath, File desinationDir,
			String fileName) {
		try {

			BufferedReader in = new BufferedReader(new FileReader(originalPath));
			PrintWriter out = getPrintWriter(desinationDir, fileName);

			String input = null;
			while ((input = in.readLine()) != null) {
				out.println(input);
			}
			in.close();
			out.close();
		} catch (FileNotFoundException ex) {
			System.out
					.println(ex.getMessage() + " in the specified directory.");
			System.exit(0);
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}

	}

	public static BufferedReader getBufferedReader(File parentDir,
			String fileName) {

		File f = new File(parentDir, fileName);
		BufferedReader reader = null;
		try {
			// Character stream I/O automatically translates Unicode to and from
			// the local character set
			// this usage is The character stream uses the byte stream to
			// perform the physical I/O,
			// while the character stream handles translation between characters
			// and bytes.
			reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(f), "UTF-8"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return reader;

	}

	public static PrintWriter getPrintWriter(File parentDir, String fileName) {
		File f = new File(parentDir, fileName);
		PrintWriter out = null;
		try {

			out = new PrintWriter(new OutputStreamWriter(
					new FileOutputStream(f), "UTF-8"));
			return out;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return out;
	}

	public static String readContentFromFile(File parentDir, String fileName) {
		File f = new File(parentDir, fileName);
		BufferedReader reader = null;
		try {
			reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(f), "UTF-8"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		final int BUFSIZE = 2048;
		char[] buf = new char[BUFSIZE];
		int count = 0;
		StringBuffer sb = new StringBuffer(BUFSIZE);
		do {
			try {
				count = reader.read(buf, 0, BUFSIZE);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if (count == -1)
				break;
			sb.append(buf, 0, count);
		} while (count == BUFSIZE);

		return new String(sb.toString());

	}

	public static void writeContentToFile(File parentDir, String fileName,
			String content) {
		try {
			File f = new File(parentDir, fileName);
			PrintWriter out = new PrintWriter(new OutputStreamWriter(
					new FileOutputStream(f), "UTF-8"));
			try {
				out.print(content);
			} finally {
				out.close();
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	public static ArrayList<File> travelFileList(File corpusDir) {
		ArrayList<File> fileList = new ArrayList<File>();
		File[] files = corpusDir.listFiles();
		Arrays.sort(files);
		for (int i = 0; i < files.length; i++) {
			if (!files[i].isDirectory()) {
				fileList.add(files[i]);
			} else {
				fileList.addAll(travelFileList(files[i]));
			}
		}
		return fileList;

	}

	public static void main(String[] args) throws IOException {

	}

}

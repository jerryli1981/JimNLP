package edu.pengli.nlp.platform.pipe;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;

import edu.pengli.nlp.platform.types.Instance;

public class Input2CharSequence extends Pipe {

	String encoding = null;

	public Input2CharSequence(String encoding) {
		this.encoding = encoding;
	}

	public Instance pipe(Instance carrier) {
		try {
			if (carrier.getData() instanceof File)
				carrier.setData(pipe((File) carrier.getData()));
			else if (carrier.getData() instanceof Reader)
				carrier.setData(pipe((Reader) carrier.getData()));
			else
				throw new IllegalArgumentException("Does not handle class "
						+ carrier.getData().getClass());
		} catch (IOException e) {
			throw new IllegalArgumentException("IOException " + e);
		}

		return carrier;
	}

	private CharSequence pipe(File file) throws IOException {
		BufferedReader br = null;
		if (encoding == null)
			try {
				br = new BufferedReader(new FileReader(file));
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		else
			try {
				br = new BufferedReader(new InputStreamReader(
						new FileInputStream(file), encoding));
			} catch (UnsupportedEncodingException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		CharSequence cs = pipe(br);
		br.close();

		return cs;

	}

	private CharSequence pipe(Reader reader) throws IOException {
		final int BUFSIZE = 2048;
		char[] buf = new char[BUFSIZE];
		int count = 0;
		StringBuffer sb = new StringBuffer(BUFSIZE);
		do {
			count = reader.read(buf, 0, BUFSIZE);
			if (count == -1)
				break;
			sb.append(buf, 0, count);

		} while (count == BUFSIZE);
		
		StringBuffer sbb = new StringBuffer();
		char[] chars = sb.toString().toCharArray();
		for (int i = 0; i < chars.length; i++) {
			char c = chars[i];
			if (String.valueOf(c).matches("\\p{Graph}")
					|| String.valueOf(c).matches("\\p{Space}")) {
				sbb.append(c);
			}
		}
		return sbb;
	}
}

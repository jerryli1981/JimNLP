/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

/** 
 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package edu.pengli.nlp.platform.pipe.iterator;

import java.io.*;
import java.util.Iterator;

import edu.pengli.nlp.platform.types.Instance;

public class OneInstancePerLineIterator implements Iterator<Instance> {

	protected BufferedReader reader = null;
	protected int index = -1;
	protected String currentLine = null;
	protected boolean hasNextUsed = false;

	public OneInstancePerLineIterator(String filename) {
		try {
			this.reader = new BufferedReader(new FileReader(filename));
			this.index = 0;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public OneInstancePerLineIterator(File file) {
		try {
			this.reader = new BufferedReader(new FileReader(file));
			this.index = 0;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public OneInstancePerLineIterator(CharSequence content){
		StringReader sReader = new StringReader((String) content);
		this.reader = new BufferedReader(sReader);
		this.index = 0;
	}

	public Instance next() {

		if (!hasNextUsed) {
			try {
				currentLine = reader.readLine();
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		} else {
			hasNextUsed = false;
		}

		return new Instance(currentLine, null, null, null);
	}

	public boolean hasNext() {
		hasNextUsed = true;
		try {
			currentLine = reader.readLine();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return (currentLine != null);
	}

	public void remove() {
		throw new IllegalStateException(
				"This Iterator<Instance> does not support remove().");
	}

}

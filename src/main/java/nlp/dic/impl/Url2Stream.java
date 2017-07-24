package nlp.dic.impl;

import nlp.dic.PathToStream;
import nlp.exception.LibraryException;

import java.io.InputStream;
import java.net.URL;

/**
 * url://http://maven.nlpcn.org/down/library/default.dic
 * 
 * @author ansj
 *
 */
public class Url2Stream extends PathToStream {

    @Override
    public InputStream toStream(String path) {
        try {
            URL url = new URL(path);
            return url.openStream();
        } catch (Exception e) {
            throw new LibraryException("err to load by http " + path + " message : " + e.getMessage());
        }

    }

}

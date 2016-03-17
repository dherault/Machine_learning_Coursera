/* global describe, it */
'use strict';

const expect = require('chai').expect;
const assert = require('chai').assert;
const Matrix = require('../lib/Matrix');

const m1 = new Matrix([
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]);
const m2 = new Matrix([
  [1, 1, 1], 
  [2, 2, 2],
]);
const m3 = new Matrix([
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
]);
const m4 = new Matrix([
  [0, 1, 1], 
  [1, 0, 1],
  [1, 1, 0],
  [1, 1, 1],
]);

describe('Matrix', () => {
  describe('constructor', () => {
    
    it('should throw on incorrect argument', () => {
      let error1, error2;
      try {
        new Matrix('yolo');
      }
      catch(err) {
        error1 = err;
      }
      try {
        new Matrix(1, 2, 3);
      }
      catch(err) {
        error2 = err;
      }
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
    });
    
    it('should accept vector shortcut argument', () => {
      const array = [1, 2, 3];
      const matrixVector = new Matrix(array);
      assert.deepEqual(matrixVector.data, [array]);
    });
    
    it('should throw on column number inconsistency', () => {
      let error;
      try {
        new Matrix([[1], [1, 2]]);
      }
      catch(err) {
        error = err;
      }
      expect(error).to.be.a('error');
    });
    
    it('should throw on non-numeric values', () => {
      let error1, error2;
      try {
        new Matrix([[1, 2], [1, undefined]]);
      }
      catch(err) {
        error1 = err;
      }
      try {
        new Matrix([[1, 2], [1, '2']]);
      }
      catch(err) {
        error2 = err;
      }
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
    });
    
    it('should have the correct dimension', () => {
      expect(m1.dimension).to.equal(9);
      expect(m2.dimension).to.equal(6);
      expect(m3.dimension).to.equal(12);
    });
  });
  
  describe('methods', () => {
    
    it('should transpose correctly', () => {
      
      assert.deepEqual(m1.transpose().data, m1.data);
      assert.deepEqual(m3.transpose().data, [
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
        [4, 8, 12],
      ]);
    });
    
    it('should throw on incorrect add argument', () => {
      
      let error1, error2, error3;
      try {
        m1.add('yolo');
      }
      catch(err) {
        error1 = err;
      }
      try {
        m1.add(1);
      }
      catch(err) {
        error2 = err;
      }
      try {
        m1.add(m2);
      }
      catch(err) {
        error3 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
      expect(error3).to.be.a('error');
    });
    
    it('should add correctly', () => {
      
      assert.deepEqual(m1.add(m1).data, [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
      ]);
      assert.deepEqual(m2.add(m2).data, [
        [2, 2, 2], 
        [4, 4, 4],
      ]);
    });
    
    
    it('should throw on incorrect multiply argument', () => {
      
      let error1, error2, error3;
      try {
        m1.multiply('yolo');
      }
      catch(err) {
        error1 = err;
      }
      try {
        m1.multiplyScalar(m1);
      }
      catch(err) {
        error2 = err;
      }
      try {
        m1.multiplyMatrix(111);
      }
      catch(err) {
        error3 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
      expect(error3).to.be.a('error');
    });
    
    it('should multiply with a scalar correctly', () => {
      
      const pi = Math.PI;
      const twoPi = 2 * pi;
      const m1_111Data = [
        [111, 0, 0],
        [0, 111, 0],
        [0, 0, 111],
      ];
      const m2_piData = [
        [pi, pi, pi],
        [twoPi, twoPi, twoPi],
      ];
      
      assert.deepEqual(m1.multiplyScalar(111).data, m1_111Data);
      assert.deepEqual(m2.multiplyScalar(pi).data, m2_piData);
      
      assert.deepEqual(m1.multiply(111).data, m1_111Data);
      assert.deepEqual(m2.multiply(pi).data, m2_piData);
    });
    
    it('should multiply with a matrix correctly', () => {
      
      const m3_m4Data = [
        [9, 8, 7],
        [21, 20, 19],
        [33, 32, 31],
      ];
      
      assert.deepEqual(m1.multiplyMatrix(m1).data, m1.data);
      assert.deepEqual(m3.multiplyMatrix(m4).data, m3_m4Data);
      
      assert.deepEqual(m1.multiply(m1).data, m1.data);
      assert.deepEqual(m3.multiply(m4).data, m3_m4Data);
    });
  });
  
});

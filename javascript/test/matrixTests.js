/* global describe, it */
'use strict';

const expect = require('chai').expect;
const assert = require('chai').assert;
const Matrix = require('../lib/Matrix');
const Vector = Matrix; // haha!
const createIdentityMatrixData = require('../lib/utils/createIdentityMatrixData');
const createRandomSquareMatrixData = require('../lib/utils/createRandomSquareMatrixData');

const identity3Data = createIdentityMatrixData(3);
const m1 = new Matrix(identity3Data);
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
const m5 = new Matrix([
  [-2, 2, -3],
  [-1, 1, 3],
  [2, 0, -1],
]);
const m6 = new Matrix([
  [4, 1, 5, 9],
  [2, -1, -6, -9],
  [0, 15, -3, 7],
  [-1, -1, 2, 0],
]);
const m7 = new Matrix([
  [0, 0], 
  [0, 0],
]);

const v1 = new Vector([1, 2, 3]); // vectors are horizontal (transposed) by default, to respect the matrix notation
const v2 = new Vector([[1], [2], [3]]); // vertical 'classic' vector
const v3 = new Vector([[10], [20], [30]]);
const v4 = new Vector([1, 2, 3, 4]);

describe('Matrix', () => {
  describe('constructor', () => {
    
    it('should accept vector shortcut argument', () => {
      assert.deepEqual(v1.data, [[1, 2, 3]]);
    });
    
    it('should have the correct dimension', () => {
      expect(m1.dimension).to.equal(9);
      expect(m2.dimension).to.equal(6);
      expect(m3.dimension).to.equal(12);
    });
    
    it('should identify square matrices', () => {
      expect(m1.isSquare).to.equal(true);
      expect(m2.isSquare).to.equal(false);
    });
    
    it('should identify vectors', () => {
      expect(v1.isVector).to.equal(true);
      expect(v2.isVector).to.equal(true);
    });
  });
  
  describe('methods', () => {
    
    it('should transpose correctly', () => {
      
      assert.deepEqual(m1.transpose().data, identity3Data);
      assert.deepEqual(m3.transpose().data, [
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
        [4, 8, 12],
      ]);
      assert.deepEqual(v1.transpose().data, v2.data);
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
      assert.deepEqual(v2.add(v3).data, [
        [11], 
        [22], 
        [33], 
      ]);
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
      const v2_111Data = [
        [111],
        [222],
        [333],
      ];
      
      assert.deepEqual(m1.multiplyScalar(111).data, m1_111Data);
      assert.deepEqual(m2.multiplyScalar(pi).data, m2_piData);
      assert.deepEqual(v2.multiplyScalar(111).data, v2_111Data);
      
      assert.deepEqual(m1.multiply(111).data, m1_111Data);
      assert.deepEqual(m2.multiply(pi).data, m2_piData);
      assert.deepEqual(v2.multiply(111).data, v2_111Data);
    });
    
    it('should multiply with a matrix correctly', () => {
      
      const m3_m4Data = [
        [9, 8, 7],
        [21, 20, 19],
        [33, 32, 31],
      ];
      const x = new Matrix([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const y = new Matrix([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
      ]);
      
      assert.deepEqual(m1.multiplyMatrix(m1).data, identity3Data);
      assert.deepEqual(m3.multiplyMatrix(m4).data, m3_m4Data);
      assert.deepEqual(m1.multiplyMatrix(v2).data, v2.data);
      
      assert.deepEqual(m1.multiply(m1).data, identity3Data);
      assert.deepEqual(m3.multiply(m4).data, m3_m4Data);
      assert.deepEqual(m1.multiply(v2).data, v2.data);
      
      assert.deepEqual(x.multiply(y).data, [
        [13, 16, 19, 22, 25],
        [27, 34, 41, 48, 55],
        [41, 52, 63, 74, 85],
      ]);
    });
    
    it('should compute the scalar product correctly', () => {
      
      expect(v1.scalarProduct(v2)).to.equal(14);
      expect(v1.scalarProduct(v3)).to.equal(140);
    });
    
    it('should compute the determinant correctly', () => {
      
      expect(m1.getDeterminant()).to.equal(1);
      expect(m5.getDeterminant()).to.equal(18); // I say so
      expect(Math.round(m6.getDeterminant())).to.equal(-1008); // Wolfram Alpha says so, -1007.9999999999995 thank you javascript...
      expect(m7.getDeterminant()).to.equal(0);
    });
    
    it('should compute the inverse correctly', () => {
      
      // Javascript numbers are complicated...
      const roundEdges = data => data.map(row => row.map(val => Math.abs(+val.toFixed(2))));
      
      assert.deepEqual(m1.inverse().data, identity3Data);
      assert.deepEqual(roundEdges(m5.inverse().multiply(m5).data), identity3Data);
      
      // stress test
      console.log('Stress test...');
      // With javascript numbers, chance to get a null determinant is... null;
      // because our determinant calculation involves division.
      const size = 200;
      const largeMatrix = new Matrix(createRandomSquareMatrixData(size));
      assert.deepEqual(roundEdges(largeMatrix.inverse().multiply(largeMatrix).data), createIdentityMatrixData(size));
    });
    
    it('should pass the ultimate test', () => {
      const x = m3
        .multiply(m4)
        .add(m1)
        .multiply(42)
        .transpose()
        .multiply(v1.transpose().add(v2))
        .transpose()
        .multiply(0.1)
        .add(new Matrix([12, 34, 56]))
        .scalarProduct(v3);
        
      expect(x).to.equal(75224); // not really a test...
    });
  });
  
  describe('errors', () => {
    
    it('should throw on incorrect constructor argument', () => {
      let error1, error2, error3, error4;
      
      try {
        new Matrix(1, 2, 3);
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      try {
        new Matrix('yolo');
      } catch(err) {
        console.log(err);
        error2 = err;
      }
      
      try {
        new Matrix([]);
      } catch(err) {
        console.log(err);
        error3 = err;
      }
      
      try {
        new Matrix([[1, 2, 3], 'yolo']);
      } catch(err) {
        console.log(err);
        error4 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
      expect(error3).to.be.a('error');
      expect(error4).to.be.a('error');
    });
    
    it('should throw on column number inconsistency', () => {
      let error;
      try {
        new Matrix([[1], [1, 2]]);
      } catch(err) {
        console.log(err);
        error = err;
      }
      
      expect(error).to.be.a('error');
    });
    
    it('should throw on non-numeric values', () => {
      let error1, error2;
      try {
        new Matrix([[1, 2], [1, undefined]]);
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      try {
        new Matrix([[1, 2], [1, '2']]);
      } catch(err) {
        console.log(err);
        error2 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
    });
    
    it('should throw on incorrect add argument', () => {
      
      let error1, error2, error3;
      try {
        m1.add(111);
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      try {
        m1.add(m2);
      } catch(err) {
        console.log(err);
        error2 = err;
      }
      
      try {
        v1.add(v4);
      } catch(err) {
        console.log(err);
        error3 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
      expect(error3).to.be.a('error');
    });
    
    it('should throw on incorrect multiply argument', () => {
      
      let error1, error2, error3, error4;
      try {
        m1.multiply('yolo');
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      try {
        m1.multiplyScalar(m1);
      } catch(err) {
        console.log(err);
        error2 = err;
      }
      
      try {
        m1.multiplyMatrix(111);
      } catch(err) {
        console.log(err);
        error3 = err;
      }
      
      try {
        m1.multiplyMatrix(m2);
      } catch(err) {
        console.log(err);
        error4 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
      expect(error3).to.be.a('error');
      expect(error4).to.be.a('error');
    });
    
    it('should throw on incorrect scalarProduct argument', () => {
      
      let error1, error2, error3, error4, error5;
      try {
        v1.scalarProduct(111);
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      try {
        v1.scalarProduct(m1);
      } catch(err) {
        console.log(err);
        error2 = err;
      }
      
      try {
        m1.scalarProduct(v1);
      } catch(err) {
        console.log(err);
        error3 = err;
      }
      
      try {
        v4.scalarProduct(v2);
      } catch(err) {
        console.log(err);
        error4 = err;
      }
      
      try {
        v2.scalarProduct(v1);
      } catch(err) {
        console.log(err);
        error5 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
      expect(error3).to.be.a('error');
      expect(error4).to.be.a('error');
      expect(error5).to.be.a('error');
    });
    
    it('should throw when a non-square matrix calls getDeterminant', () => {
      
      let error1;
      try {
        m2.getDeterminant();
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      expect(error1).to.be.a('error');
    });
    
    it('should throw when calling inverse with bad conditions', () => {
      
      let error1, error2;
      
      try {
        m2.inverse(); // Not square
      } catch(err) {
        console.log(err);
        error1 = err;
      }
      
      try {
        m7.inverse(); // Det === 0
      } catch(err) {
        console.log(err);
        error2 = err;
      }
      
      expect(error1).to.be.a('error');
      expect(error2).to.be.a('error');
    });
  });
});
